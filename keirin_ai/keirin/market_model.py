"""オッズだけ（＋任意でライン連動のAI確率）から各組み合わせの的中確率を出し、1レース1点の買い目を決める。

- 4券種（2車複・ワイド・2車単・3連複）のオッズを、それぞれ「市場の確率」（1/オッズ をレース内で合計1）にする。
- 券種どうしのずれ（例: 2車単から見た 1-2 の確率 と 2車複の 1-2 の確率の違い）を特徴にして、
  ロジスティック回帰で「その目が当たる確率」を出す。係数は過去の確定オッズで当てはめたもの（data/ に置く。git に入れない）。
- ルール: 30倍以下の全券種の目のうち、期待値（確率×オッズ）が一番高い目を、基準以上なら1点買う。

学習（9/09〜10/14）→確認（10/15〜11/19、確定オッズ）: 期待値1.1以上 98.8%、1.2以上 110.5%（オッズだけ）。
ライン連動のAI確率を足すと 104.9% / 113.6%（差は誤差の範囲）。締切前オッズでの成績はこれから記録して確かめる。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BTS = ("2車複", "ワイド", "2車単", "3連複")


def _pair(c: str) -> str:
    return "-".join(sorted(c.replace(">", "-").split("-"), key=int))


def market_table(odds_by_type: dict[str, dict[str, float]], n_riders: int,
                 line_probs: dict[str, dict[str, float]] | None = None) -> pd.DataFrame | None:
    """→ 1行1組み合わせ（bt, combo, odds と特徴の列）。4券種そろっていなければ None。"""
    if any(not odds_by_type.get(bt) for bt in BTS):
        return None
    q = {}
    for bt in BTS:
        inv = {c: 1.0 / o for c, o in odds_by_type[bt].items() if o and o > 0}
        s = sum(inv.values())
        q[bt] = {c: v / s for c, v in inv.items()}
    q_ex, q_tr, pw = {}, {}, {}
    for c, v in q["2車単"].items():
        q_ex[_pair(c)] = q_ex.get(_pair(c), 0.0) + v
        f = c.split(">")[0]
        pw[f] = pw.get(f, 0.0) + v
    for c, v in q["3連複"].items():
        a, b, d = c.split("-")
        for k in (f"{a}-{b}", f"{a}-{d}", f"{b}-{d}"):
            q_tr[k] = q_tr.get(k, 0.0) + v
    q_qu, q_wi = q["2車複"], q["ワイド"]
    rows = []
    for bt in BTS:
        for c, o in odds_by_type[bt].items():
            if c not in q[bt]:
                continue
            r = {"bt": bt, "combo": c, "odds": float(o), "q": q[bt][c], "nine": float(n_riders == 9)}
            if bt != "3連複":
                k = _pair(c)
                r.update(q_ex=q_ex.get(k, np.nan), q_tr=q_tr.get(k, np.nan), q_qu=q_qu.get(k, np.nan), q_wi=q_wi.get(k, np.nan))
                if bt == "2車単":
                    a, b = c.split(">")
                    rv = q["2車単"].get(f"{b}>{a}", np.nan)
                    r["share"] = r["q"] / (r["q"] + rv)
            else:
                a, b, d = c.split("-")
                ps = (f"{a}-{b}", f"{a}-{d}", f"{b}-{d}")
                r.update(s_wi=sum(q_wi.get(k, np.nan) for k in ps), s_qu=sum(q_qu.get(k, np.nan) for k in ps),
                         s_ex=sum(q_ex.get(k, np.nan) for k in ps),
                         w3=float(np.prod([pw.get(x, np.nan) for x in (a, b, d)])))
            if line_probs is not None:
                r["pl"] = line_probs.get(bt, {}).get(c, np.nan)
            rows.append(r)
    return pd.DataFrame(rows)


def apply_model(tab: pd.DataFrame, model: dict) -> pd.DataFrame:
    """model: {券種: {cols, mu, sd, w}}（学習スクリプトの出力）。→ pm（当たる確率）と ev 列を足す。"""
    out = []
    for bt, x in tab.groupby("bt"):
        m = model.get(bt)
        if m is None:
            continue
        x = x.copy()
        cols = list(m["cols"])
        if any(c not in x for c in cols):
            continue
        F = np.column_stack([np.log(x[c].astype(float).clip(lower=1e-9)) for c in cols]
                            + [np.log(x["odds"].astype(float)), x["nine"].astype(float)])
        Z = np.column_stack([np.ones(len(F)), (F - np.asarray(m["mu"])) / np.asarray(m["sd"])])
        x["pm"] = 1.0 / (1.0 + np.exp(-(Z @ np.asarray(m["w"]))))
        x = x[np.isfinite(x["pm"])]
        x["ev"] = x["pm"] * x["odds"]
        out.append(x)
    return pd.concat(out) if out else tab.iloc[0:0]


def one_bet(tab: pd.DataFrame, min_ev: float, max_odds: float = 30.0) -> dict | None:
    """30倍以下の目のうち期待値が一番高い1点。基準未満なら None（見送り）。"""
    t = tab[tab["odds"] <= max_odds]
    if t.empty:
        return None
    best = t.loc[t["ev"].idxmax()]
    return best.to_dict() if best["ev"] >= min_ev else None


__all__ = ["market_table", "apply_model", "one_bet", "BTS"]
