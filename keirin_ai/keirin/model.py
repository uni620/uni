"""予想方法（設計書「比べる相手」の表）。

- 手作り7:3       : 強さ7・調子3 の重みで点数をつける（AIなし）
- LightGBM        : 設計書の列だけで学習
- LightGBM+競走得点: 上に競走得点を足して学習（「競走得点は役に立たない」を確かめる）
- 人気順          : オッズから逆算（strategy.py の「人気1位」と、評価の的中率で使う）

どの方法も「各選手の1着確率（レース内で合計100%）」を出す。
LightGBM はさらに「3着以内の確率」「ライン2人がそろって2着以内の確率」も出す。
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .features import BANK_FEATURES, CARD_FEATURES, CATEGORICAL, FEATURE_JA, FEATURES, POS_RECORD_FEATURES, SCORE_FEATURE

TARGETS = {"win": "y_win", "top3": "y_top3", "pair": "y_pair"}

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
}


def normalize_in_race(p: pd.Series, race_id: pd.Series) -> pd.Series:
    """各選手の確率を、レース内で合計1になるように直す。"""
    p = p.clip(1e-6, None)
    return p / p.groupby(race_id).transform("sum")


def _softmax_in_race(u: pd.Series, race_id: pd.Series) -> pd.Series:
    u = u - u.groupby(race_id).transform("max")
    e = np.exp(u)
    return e / e.groupby(race_id).transform("sum")


def _z_in_race(x: pd.Series, race_id: pd.Series) -> pd.Series:
    m = x.groupby(race_id).transform("mean")
    s = x.groupby(race_id).transform("std").replace(0, np.nan)
    return ((x - m) / s).fillna(0.0)


class HandScore:
    """手作り7:3: 点数 = 0.7×強さ + 0.3×調子（どちらもレース内で標準化）。
    点数→確率の変換の強さ（温度）だけは学習データに合わせる。"""
    name = "手作り7:3"

    def __init__(self, w_strength: float = 0.7, w_form: float = 0.3):
        self.w_s, self.w_f = w_strength, w_form
        self.temp = 1.0

    def score(self, df: pd.DataFrame) -> pd.Series:
        strength = _z_in_race(-df["recent_finish"], df["race_id"])
        form = _z_in_race(df["top3_3m"], df["race_id"])
        return self.w_s * strength + self.w_f * form

    def fit(self, df: pd.DataFrame) -> "HandScore":
        d = df[df["y_win"].notna()]
        s = self.score(d)
        best, best_ll = 1.0, np.inf
        for t in np.linspace(0.1, 3.0, 30):
            p = _softmax_in_race(s * t, d["race_id"])
            ll = -np.log(p[d["y_win"] == 1].clip(1e-9)).mean()
            if ll < best_ll:
                best, best_ll = t, ll
        self.temp = float(best)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        p = _softmax_in_race(self.score(df) * self.temp, df["race_id"])
        return pd.DataFrame({"p_win": p}, index=df.index)

    def importance(self) -> pd.DataFrame:
        return pd.DataFrame({"特徴量": ["強さ（直近の大会の着順）", "調子（3か月の複勝率）"],
                             "重要度": [self.w_s, self.w_f]})


class LGBModel:
    """1着・3着以内・ライン2人そろって2着以内 の3つを、それぞれ LightGBM で学習する。"""

    def __init__(self, use_score: bool = False, params: dict | None = None, num_rounds: int = 2000,
                 pos_bank: bool = False, card: bool = False):
        self.use_score = use_score
        self.name = "LightGBM" + ("+競走得点" if use_score else "") + ("+位置バンク" if pos_bank else "") + ("+出走表" if card else "")
        # pos_bank: 「同じ位置・同じバンクでの過去の成績」も使う版（試験中。本番の仮想購入で並べて比べる）
        self.features = (FEATURES + ([SCORE_FEATURE] if use_score else [])
                         + (POS_RECORD_FEATURES + BANK_FEATURES if pos_bank else [])
                         + (CARD_FEATURES if card else []))
        self.params = {**LGB_PARAMS, **(params or {})}
        self.num_rounds = num_rounds
        self.boosters: dict = {}
        self.best_iter: dict[str, int] = {}
        self.version = ""
        self.trained_races = 0

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.features].astype(float)

    def fit(self, df: pd.DataFrame, valid_frac: float = 0.15) -> "LGBModel":
        import lightgbm as lgb

        d = df[df["y_win"].notna()].copy()
        self.trained_races = int(d["race_id"].nunique())
        # 学習データのうち新しい方の 15% で「何回学習を重ねるか」を決める（時間で分ける）
        dates = np.sort(d["date"].unique())
        cut = dates[int(len(dates) * (1 - valid_frac))] if len(dates) > 5 else dates[-1]
        for key, ycol in TARGETS.items():
            dd = d[d[ycol].notna()]
            if dd[ycol].nunique() < 2:
                continue
            tr, va = dd[dd["date"] < cut], dd[dd["date"] >= cut]
            cat = [c for c in CATEGORICAL if c in self.features]
            dtr = lgb.Dataset(self._X(tr), tr[ycol], categorical_feature=cat, free_raw_data=False)
            if len(va) and va[ycol].nunique() == 2:
                dva = lgb.Dataset(self._X(va), va[ycol], reference=dtr, categorical_feature=cat)
                b = lgb.train(self.params, dtr, self.num_rounds, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(100, verbose=False)])
                best = max(50, b.best_iteration or 200)
            else:
                best = 300
            # 決めた回数で、学習データ全部を使って学習し直す
            dall = lgb.Dataset(self._X(dd), dd[ycol], categorical_feature=cat)
            self.boosters[key] = lgb.train(self.params, dall, best)
            self.best_iter[key] = best
        self.version = datetime.now().strftime("%Y%m%d%H%M%S")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._X(df)
        out = pd.DataFrame(index=df.index)
        raw_win = pd.Series(self.boosters["win"].predict(X), index=df.index)
        out["p_win"] = normalize_in_race(raw_win, df["race_id"])
        if "top3" in self.boosters:
            out["p_top3"] = self.boosters["top3"].predict(X)
        if "pair" in self.boosters:
            pair_rows = (df["line_size"].fillna(1) >= 2) & (df["line_pos"] <= 2)
            out["p_pair"] = np.where(pair_rows, self.boosters["pair"].predict(X), np.nan)
        return out

    def importance(self) -> pd.DataFrame:
        if "win" not in self.boosters:
            return pd.DataFrame(columns=["特徴量", "重要度"])
        b = self.boosters["win"]
        imp = b.feature_importance("gain")
        df = pd.DataFrame({"特徴量": [FEATURE_JA.get(f, f) for f in b.feature_name()], "重要度": imp})
        df["重要度"] = df["重要度"] / max(df["重要度"].sum(), 1e-9)
        return df.sort_values("重要度", ascending=False)


def make_methods(names: list[str]) -> list:
    table = {"手作り7:3": lambda: HandScore(), "LightGBM": lambda: LGBModel(False),
             "LightGBM+競走得点": lambda: LGBModel(True),
             "LightGBM+競走得点+位置バンク": lambda: LGBModel(True, pos_bank=True),
             "LightGBM+競走得点+位置バンク+出走表": lambda: LGBModel(True, pos_bank=True, card=True)}
    return [table[n]() for n in names if n in table]


def evaluate(df: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """的中率などの成績（設計書「評価の順番」の1番目）。"""
    d = df.join(pred[[c for c in pred.columns if c not in df.columns]])
    d = d[d["finish"].notna() & d["p_win"].notna()]
    if d.empty:
        return {}
    top = d.loc[d.groupby("race_id")["p_win"].idxmax()]
    winners = d[d["finish"] == 1]
    n = d.groupby("race_id")["car_no"].transform("count")
    out = {
        "検証レース数": int(top["race_id"].nunique()),
        "1位予想の1着率": float((top["finish"] == 1).mean()),
        "1位予想の3着以内率": float((top["finish"] <= 3).mean()),
        "予想のズレ(小さいほど良い)": float(-np.log(winners["p_win"].clip(1e-9)).mean()),
        "でたらめ予想のズレ": float(-np.log(1 / n[winners.index]).mean()),
    }
    if "p_pair" in d and d["p_pair"].notna().any():
        pr = d[d["p_pair"].notna()].copy()
        best = pr.loc[pr.groupby("race_id")["p_pair"].idxmax()]
        out["最有力ライン2人で2着以内を独占した率"] = float(best["y_pair"].mean())
    return out
