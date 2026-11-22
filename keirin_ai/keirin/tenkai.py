"""展開AI v1（2026-11-22 固定）: 強さの数字（競走得点・勝率・連対率）を使わず、展開だけで予想する。

段階1: 誰が B（バック先頭＝主導権）を取るか（戦法の割合・位置・車番・先行争いの人数）。
段階2: その展開で誰が1着か（段階1の確率・自分のラインの確率・他ラインの最大・位置・戦法）。
学習 1/1〜8/31 → 確認 9/1〜11/21: 主導権の的中 62.6%（でたらめなら14%）、本命の1着率 38.1%。
人気に混ぜたときの比重は 0.12（今のAIは 0.03）。上乗せは +0.75‰ で、まだ偶然と区別できない。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

A = ["逃率", "捲率", "差率", "マ率", "S率", "B率", "H率", "自力率", "単騎", "先頭", "番手", "3番手+", "ライン人数",
     "car_no", "n", "ライン数", "先行候補数", "B率の順位", "レース最大B率"]
B = A + ["pB", "自分のラインのpB", "自分の先頭のpB", "他ラインの最大pB", "競り合い"]
P = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=50, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=1)
QUERY = """SELECT e.race_id, e.car_no, e.line_id, e.line_pos, e.line_size, e.scratched, e.finish, e.km_nige, e.km_makuri,
  e.km_sashi, e.km_mark, e.sb_s, e.sb_h, e.sb_b, e.n1, e.n2, e.n3, e.nout, r.date, r.race_class, i.sb
  FROM entries e JOIN races r USING(race_id) LEFT JOIN result_info i USING(race_id, car_no)
  WHERE r.race_class NOT LIKE '%ガールズ%'"""


def features(E: pd.DataFrame) -> pd.DataFrame:
    E = E.copy()
    for k in ["km_nige", "km_makuri", "km_sashi", "km_mark", "sb_s", "sb_h", "sb_b", "n1", "n2", "n3", "nout"]:
        E[k] = pd.to_numeric(E[k], errors="coerce").fillna(0)
    km = E[["km_nige", "km_makuri", "km_sashi", "km_mark"]].sum(axis=1).replace(0, np.nan)
    st = E[["n1", "n2", "n3", "nout"]].sum(axis=1).replace(0, np.nan)
    for k, lab in (("km_nige", "逃"), ("km_makuri", "捲"), ("km_sashi", "差"), ("km_mark", "マ")):
        E[lab + "率"] = E[k] / km
    E["S率"], E["B率"], E["H率"] = E.sb_s / st, E.sb_b / st, E.sb_h / st
    E["自力率"] = E["逃率"].fillna(0) + E["捲率"].fillna(0)
    sz, ps = E.line_size.fillna(1), E.line_pos.fillna(1)
    E["単騎"] = (sz <= 1).astype(float); E["先頭"] = ((sz >= 2) & (ps == 1)).astype(float)
    E["番手"] = ((sz >= 2) & (ps == 2)).astype(float); E["3番手+"] = ((sz >= 2) & (ps >= 3)).astype(float)
    E["ライン人数"] = sz
    E["lk"] = E.race_id + "_" + E.line_id.fillna(-E.car_no).astype(int).astype(str)
    E["n"] = E.groupby("race_id").car_no.transform("size"); E["ライン数"] = E.groupby("race_id").lk.transform("nunique")
    ld = E[(E["先頭"] == 1) | (E["単騎"] == 1)]
    E["先行候補数"] = E.race_id.map(ld.assign(f=(ld["B率"] > 0.2) | (ld["逃率"] > 0.4)).groupby("race_id").f.sum())
    E["レース最大B率"] = E.race_id.map(ld.groupby("race_id")["B率"].max())
    E["B率の順位"] = E.groupby("race_id")["B率"].rank(ascending=False)
    return E


def stage2_features(E: pd.DataFrame) -> pd.DataFrame:
    E = E.copy()
    E["pB"] = E.pB / E.groupby("race_id").pB.transform("sum")
    E["自分のラインのpB"] = E.groupby("lk").pB.transform("sum")
    E["自分の先頭のpB"] = E.lk.map(E[E["先頭"] == 1].set_index("lk").pB).fillna(0)
    L = E.groupby(["race_id", "lk"]).pB.sum().rename("lp").reset_index()
    L["m1"] = L.groupby("race_id").lp.transform("max")
    L["m2"] = L.groupby("race_id").lp.transform(lambda s: s.nlargest(2).iloc[-1] if len(s) > 1 else 0)
    L["other"] = np.where(L.lp >= L.m1, L.m2, L.m1)
    E["他ラインの最大pB"] = E.lk.map(L.set_index("lk").other)
    E["競り合い"] = E.groupby("race_id").pB.transform(lambda s: 1 - s.max())
    return E


def train_and_predict(con, today: str) -> pd.DataFrame:
    """today より前の結果で学習し、today のレースを予想 → race_id, car_no, pB（主導権）, p_win。"""
    import lightgbm as lgb
    E = features(pd.read_sql(QUERY, con))
    E["finish"] = pd.to_numeric(E.finish, errors="coerce")
    scr = set(E[E.scratched == 1].race_id)
    hist = E[(E.date < today) & ~E.race_id.isin(scr)].copy()
    hist = hist[hist.groupby("race_id").finish.transform(lambda s: (s == 1).sum() == 1)]
    hist["gotB"] = hist.sb.astype(str).str.contains("B").astype(float)
    hist = hist[hist.groupby("race_id").gotB.transform("sum") == 1]
    tgt = E[E.date == today].copy()
    if tgt.empty or hist.empty:
        return pd.DataFrame(columns=["race_id", "car_no", "pB", "p_win"])
    # 段階1: 学習データの中は2つに分けて交互に（段階2の学習に、自分の答えを見ていない確率を使う）
    hist = hist.sort_values("date")
    half = hist.date < hist.date.iloc[len(hist) // 2]
    hist["pB"] = np.nan
    for a, b in ((half, ~half), (~half, half)):
        m = lgb.train(P, lgb.Dataset(hist.loc[a, A], hist.loc[a, "gotB"]), 300)
        hist.loc[b, "pB"] = m.predict(hist.loc[b, A])
    m1 = lgb.train(P, lgb.Dataset(hist[A], hist.gotB), 300)
    tgt["pB"] = m1.predict(tgt[A])
    hist, tgt = stage2_features(hist), stage2_features(tgt)
    m2 = lgb.train(P, lgb.Dataset(hist[B], (hist.finish == 1).astype(float)), 400)
    tgt["p_win"] = m2.predict(tgt[B])
    tgt["p_win"] = tgt.p_win / tgt.groupby("race_id").p_win.transform("sum")
    return tgt[["race_id", "car_no", "pB", "p_win"]]
