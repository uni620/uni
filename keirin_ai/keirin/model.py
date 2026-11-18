"""選手ごとの1着確率を出すAI（条件付きロジット / Plackett-Luce モデル）。

考え方
- 各選手に「強さスコア u = 特徴量 × 重み」をつける。
- そのレースでの1着確率 = exp(u_i) / Σ exp(u_j) （レース内で合計100%になる）。
- 学習では、実際の1着・2着・3着が起きる確率が最大になるように重みを決める。
  （1着を当てたあと、残りの選手で2着を当てる…という考え方 = Plackett-Luce）
- 香港競馬で成功した Bill Benter の手法の土台と同じ形。中身が読める（重みが見える）のが利点。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .features import FEATURES

FEATURE_JA = {
    "score_z": "競走得点（レース内で高いほど+）", "score_rank": "競走得点の順位（下位ほど+）",
    "class_diff": "級班（レース内で上位ほど+）", "is_nige": "脚質=逃", "is_ryo": "脚質=両",
    "is_line_head": "ライン先頭", "is_bante": "番手（ライン2番目）", "is_third_plus": "ライン3番手以降",
    "is_tanki": "単騎", "line_strength": "ラインの強さ（平均得点）", "line_size": "ラインの人数",
    "head_nige": "ライン先頭かつ逃", "home": "地元", "age_diff": "年齢（レース内で上ほど+）",
    "form_win": "直近の1着率", "form_top2": "直近の2着内率", "form_top3": "直近の3着内率",
    "form_n": "直近の出走数",
}


class WinProbModel:
    def __init__(self, l2: float = 1.0, depth: int = 3, weights: list[float] | None = None):
        self.l2 = l2
        self.depth = depth
        self.weights = list(weights or [1.0, 0.7, 0.5])[:depth]
        self.coef: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.version = ""
        self.trained_races = 0
        self.metrics: dict = {}

    # ---------- 学習 ----------
    def _prepare(self, df: pd.DataFrame):
        df = df.sort_values(["race_id", "car_no"])
        X = df[FEATURES].to_numpy(float)
        X = (X - self.mu) / self.sd
        _, starts = np.unique(df["race_id"].to_numpy(), return_index=True)
        starts = np.sort(starts)
        ends = np.append(starts[1:], len(df))
        fin = df["finish"].to_numpy(float)
        return X, list(zip(starts, ends)), fin

    def fit(self, df: pd.DataFrame) -> "WinProbModel":
        df = df[df["finish"].notna()].copy()
        # 1着が決まっているレースだけ使う
        has_win = df.groupby("race_id")["finish"].transform(lambda s: (s == 1).any())
        df = df[has_win]
        self.mu = df[FEATURES].mean().to_numpy(float)
        self.sd = df[FEATURES].std().replace(0, 1).fillna(1).to_numpy(float)
        X, spans, fin = self._prepare(df)
        self.trained_races = len(spans)

        # 各レース・各段階（1着, 2着, 3着）の「残っている選手」と「その段階の勝者」
        stages = []  # (idx_array_of_remaining, winner_local_index, weight)
        for s, e in spans:
            remaining = list(range(s, e))
            for d in range(min(self.depth, e - s - 1)):
                target_pos = d + 1
                hit = [i for i in remaining if fin[i] == target_pos]
                if len(hit) != 1:
                    break  # 同着・失格などで順位が曖昧ならそこで打ち切り
                w = hit[0]
                stages.append((np.array(remaining), remaining.index(w), self.weights[d]))
                remaining.remove(w)

        k = X.shape[1]
        if not stages:
            raise ValueError("学習に使える着順データがありません")
        # 計算を速くするため、段階ごとの候補を「段階数 × 最大人数」の表に詰める
        width = max(len(idx) for idx, _, _ in stages)
        IDX = np.zeros((len(stages), width), dtype=int)
        MASK = np.zeros((len(stages), width), dtype=bool)
        WIN = np.zeros(len(stages), dtype=int)
        WT = np.zeros(len(stages))
        for r, (idx, wi, wt) in enumerate(stages):
            IDX[r, : len(idx)] = idx
            MASK[r, : len(idx)] = True
            WIN[r] = idx[wi]
            WT[r] = wt
        XW = X[WIN]                    # 各段階の勝者の特徴量
        XC = X[IDX] * MASK[..., None]  # 候補の特徴量（空き枠は0）

        def loss_grad(beta):
            u = np.where(MASK, X[IDX] @ beta, -np.inf)
            m = u.max(axis=1, keepdims=True)
            ex = np.exp(u - m)
            z = ex.sum(axis=1, keepdims=True)
            p = ex / z
            logp_win = XW @ beta - m[:, 0] - np.log(z[:, 0])
            loss = -(WT * logp_win).sum() + 0.5 * self.l2 * beta @ beta
            expected = np.einsum("sw,swk->sk", p, XC)
            grad = -(WT[:, None] * (XW - expected)).sum(axis=0) + self.l2 * beta
            return loss, grad

        res = minimize(loss_grad, np.zeros(k), jac=True, method="L-BFGS-B")
        self.coef = res.x
        self.version = datetime.now().strftime("%Y%m%d%H%M%S")
        return self

    # ---------- 予測 ----------
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """1行=1選手。レース内で合計1になる1着確率を返す（index は df と同じ）。"""
        if self.coef is None:
            return self.predict_baseline(df)
        X = (df[FEATURES].to_numpy(float) - self.mu) / self.sd
        u = pd.Series(X @ self.coef, index=df.index)
        u = u - u.groupby(df["race_id"]).transform("max")
        ex = np.exp(u)
        return ex / ex.groupby(df["race_id"]).transform("sum")

    @staticmethod
    def predict_baseline(df: pd.DataFrame) -> pd.Series:
        """学習前の仮モデル: 競走得点だけで決める。"""
        u = pd.Series(df["score_z"].to_numpy(float) * 1.5, index=df.index)
        ex = np.exp(u - u.groupby(df["race_id"]).transform("max"))
        return ex / ex.groupby(df["race_id"]).transform("sum")

    # ---------- 評価 ----------
    @staticmethod
    def evaluate(df: pd.DataFrame, probs: pd.Series) -> dict:
        d = df.assign(p=probs)
        d = d[d["finish"].notna()]
        winners = d[d["finish"] == 1]
        if winners.empty:
            return {}
        n_riders = d.groupby("race_id")["car_no"].transform("count")
        uni = (1.0 / n_riders)[winners.index]
        ll_model = -np.log(winners["p"].clip(1e-9)).mean()
        ll_uniform = -np.log(uni).mean()
        top = d.loc[d.groupby("race_id")["p"].idxmax()]
        return {
            "検証レース数": int(len(winners)),
            "予想のズレ(AI, 小さいほど良い)": float(ll_model),
            "予想のズレ(でたらめ予想)": float(ll_uniform),
            "改善度(0=でたらめ並み)": float(1 - ll_model / ll_uniform),
            "AI1位の1着率": float((top["finish"] == 1).mean()),
        }

    # ---------- 保存 ----------
    def save(self, path: str | Path):
        data = {
            "features": FEATURES, "coef": self.coef.tolist() if self.coef is not None else None,
            "mu": self.mu.tolist() if self.mu is not None else None,
            "sd": self.sd.tolist() if self.sd is not None else None,
            "l2": self.l2, "depth": self.depth, "weights": self.weights,
            "version": self.version, "trained_races": self.trained_races, "metrics": self.metrics,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "WinProbModel":
        m = cls()
        p = Path(path)
        if not p.exists():
            return m
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("features") != FEATURES or data.get("coef") is None:
            return m  # 特徴量の構成が変わっていたら作り直し
        m.coef = np.array(data["coef"])
        m.mu = np.array(data["mu"])
        m.sd = np.array(data["sd"])
        m.l2, m.depth, m.weights = data["l2"], data["depth"], data["weights"]
        m.version, m.trained_races = data["version"], data["trained_races"]
        m.metrics = data.get("metrics", {})
        return m

    def coef_table(self) -> pd.DataFrame:
        if self.coef is None:
            return pd.DataFrame(columns=["特徴量", "重み", "意味"])
        df = pd.DataFrame({"特徴量": [FEATURE_JA.get(f, f) for f in FEATURES], "重み": self.coef})
        df["意味"] = np.where(df["重み"] > 0, "勝ちやすくなる", "勝ちにくくなる")
        return df.sort_values("重み", key=abs, ascending=False)
