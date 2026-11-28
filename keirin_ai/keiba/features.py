"""単勝予想用の特徴量（個人の分析用）。

考え方: 「馬の強さ」（過去走のタイム指数・着順・人気・上がり）と
「条件の有利不利」（枠・脚質・距離変化・馬場状態・コース）を分けて作り、勾配ブースティングで勝率を学習する。
過去走の情報はすべて「そのレースより前の日付」だけを使う（未来の情報を混ぜない）。
"""
from __future__ import annotations

import re
import sqlite3

import numpy as np
import pandas as pd

GRADE = [("GI)", 7), ("(GI", 7), ("GII", 6), ("GIII", 5), ("(L)", 4), ("OP", 4), ("3勝", 3), ("1600万", 3),
         ("2勝", 2), ("1000万", 2), ("1勝", 1), ("500万", 1), ("新馬", 0), ("未勝利", 0)]
GOING = {"良": 0, "稍": 1, "稍重": 1, "重": 2, "不": 3, "不良": 3}


def class_level(name: str, cls: str) -> float:
    s = f"{name or ''} {cls or ''}".translate(str.maketrans("０１２３４５６７８９", "0123456789")).replace("オープン", "OP")
    if "(GIII)" in s or "GIII" in s:
        return 5
    if "(GII)" in s:
        return 6
    if "(GI)" in s:
        return 7
    for k, v in GRADE:
        if k in s:
            return v
    return np.nan


def _sec(t):
    m = re.match(r"(\d+):(\d+\.\d)", str(t))
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    try:
        return float(t)
    except (TypeError, ValueError):
        return np.nan


def waku_of(umaban: int, n: int) -> int:
    """JRAの枠の決め方: 8頭以下は1頭1枠、それ以上は外の枠から2頭（3頭）になる。"""
    if n <= 8:
        return umaban
    base, extra = divmod(n, 8)        # extra 個の枠が base+1 頭
    sizes = [base + (1 if i >= 8 - extra else 0) for i in range(8)]
    c = 0
    for i, s in enumerate(sizes):
        c += s
        if umaban <= c:
            return i + 1
    return 8


def load(db: str) -> pd.DataFrame:
    con = sqlite3.connect(db)
    q = """SELECT r.race_id, r.date, r.venue, r.surface, r.distance, r.going, r.race_name, r.class_text, r.n_horses,
                  r.direction, x.umaban, x.waku, x.finish_text, x.horse_id, x.sex, x.age, x.weight_carried, x.jockey_id,
                  x.trainer_id, x.time, x.passing, x.last3f, x.win_odds, x.popularity, x.body_weight, x.body_weight_diff
           FROM k_races r JOIN k_results x USING(race_id)"""
    df = pd.read_sql(q, con)
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    for c in ["umaban", "waku", "age", "weight_carried", "win_odds", "popularity", "body_weight"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["fin"] = pd.to_numeric(d["finish_text"], errors="coerce")
    d["ran"] = d["fin"].notna()
    d["n"] = d.groupby("race_id")["umaban"].transform("count")
    d["win"] = (d["fin"] == 1).astype(int)
    d["sec"] = d["time"].map(_sec)
    d["l3f"] = pd.to_numeric(d["last3f"], errors="coerce")
    d["pos1"] = pd.to_numeric(d["passing"].fillna("").str.split("-").str[0], errors="coerce")
    d["pos_last"] = pd.to_numeric(d["passing"].fillna("").str.split("-").str[-1], errors="coerce")
    d["going_n"] = d["going"].map(lambda g: GOING.get(str(g)[:2], GOING.get(str(g)[:1], np.nan)))
    d["cls"] = [class_level(a, b) for a, b in zip(d["race_name"], d["class_text"])]
    d["course"] = d["venue"].fillna("") + d["surface"].fillna("") + d["distance"].astype("Int64").astype(str)
    d["sex_n"] = d["sex"].map({"牡": 0, "牝": 1, "セ": 2})
    d["bwd"] = pd.to_numeric(d["body_weight_diff"], errors="coerce")
    # タイム指数: コース×馬場の勝ちタイム中央値との差を 1000m あたりの秒で（大きいほど速い）
    w = d[d["fin"] == 1]
    std = w.groupby(["course", "going_n"])["sec"].median().rename("std_sec")
    d = d.join(std, on=["course", "going_n"])
    d["fig"] = (d["std_sec"] - d["sec"]) / d["distance"] * 1000 + d["cls"].fillna(2) * 0.3
    d["fin_r"] = (d["fin"] - 1) / (d["n"] - 1).clip(lower=1)
    d["pos1_r"] = (d["pos1"] - 1) / (d["n"] - 1).clip(lower=1)
    d["l3f_rank"] = d.groupby("race_id")["l3f"].rank(method="min")
    d["l3f_r"] = (d["l3f_rank"] - 1) / (d["n"] - 1).clip(lower=1)
    d["lodds"] = np.log(d["win_odds"])
    return d


HIST = ["fig", "fin_r", "lodds", "l3f_r", "pos1_r", "popularity"]


def history_features(d: pd.DataFrame) -> pd.DataFrame:
    """各出走について、その馬の「それより前の走」から作る特徴。"""
    d = d.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    g = d.groupby("horse_id", sort=False)
    out = pd.DataFrame(index=d.index)
    for c in HIST:
        s = g[c].shift(1)
        out[f"{c}_p1"] = s
        out[f"{c}_m3"] = s.groupby(d["horse_id"]).transform(lambda x: x.rolling(3, min_periods=1).mean())
        out[f"{c}_m5"] = s.groupby(d["horse_id"]).transform(lambda x: x.rolling(5, min_periods=1).mean())
    sfig = g["fig"].shift(1)
    out["fig_max5"] = sfig.groupby(d["horse_id"]).transform(lambda x: x.rolling(5, min_periods=1).max())
    out["n_runs"] = g.cumcount()
    out["n_wins"] = g["win"].transform(lambda x: x.shift(1).fillna(0).cumsum())
    out["days"] = (d["date"] - g["date"].shift(1)).dt.days
    out["dist_p1"] = g["distance"].shift(1)
    out["dist_chg"] = d["distance"] - out["dist_p1"]
    out["surf_chg"] = (d["surface"] != g["surface"].shift(1)).astype(float).where(out["n_runs"] > 0)
    out["cls_p1"] = g["cls"].shift(1)
    out["cls_chg"] = d["cls"] - out["cls_p1"]
    # 同じ芝/ダでのタイム指数（過去5走平均）
    key = d["horse_id"] + d["surface"].fillna("")
    out["fig_same_surf"] = d.groupby(key)["fig"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    out["wc_chg"] = d["weight_carried"] - g["weight_carried"].shift(1)
    return pd.concat([d, out], axis=1)


def rate_features(d: pd.DataFrame, key: str, name: str, prior: float = 0.07, k: float = 30) -> pd.DataFrame:
    """騎手・調教師の、その日より前の勝率（少ない数は全体平均に寄せる）。"""
    d = d.sort_values(["date", "race_id"]).reset_index(drop=True)
    day = d.groupby([key, "date"]).agg(w=("win", "sum"), n=("win", "size")).reset_index().sort_values([key, "date"])
    day["cw"] = day.groupby(key)["w"].cumsum() - day["w"]
    day["cn"] = day.groupby(key)["n"].cumsum() - day["n"]
    day[name] = (day["cw"] + prior * k) / (day["cn"] + k)
    day[name + "_n"] = day["cn"]
    return d.merge(day[[key, "date", name, name + "_n"]], on=[key, "date"], how="left")


FEATURES = ([f"{c}_{s}" for c in HIST for s in ("p1", "m3", "m5")] +
            ["fig_max5", "n_runs", "n_wins", "days", "dist_chg", "surf_chg", "cls_chg", "fig_same_surf", "wc_chg",
             "jockey_rate", "jockey_rate_n", "trainer_rate", "trainer_rate_n",
             "waku", "umaban", "n", "distance", "going_n", "cls", "age", "sex_n", "weight_carried",
             "venue_c", "surface_c", "course_c", "direction_c"])
CATS = ["venue_c", "surface_c", "course_c", "direction_c"]


def add_relative(d: pd.DataFrame) -> pd.DataFrame:
    """同じレースの中での比較（横の比較）: 強さの指標をレース内で順位化・平均との差にする。"""
    for c in ["fig_m3", "fig_max5", "fig_p1", "lodds_m3", "fin_r_m3", "jockey_rate", "fig_same_surf"]:
        m = d.groupby("race_id")[c].transform("mean")
        d[c + "_rel"] = d[c] - m
        d[c + "_rk"] = d.groupby("race_id")[c].rank(ascending=c.startswith("lodds") or c.startswith("fin_r"), pct=True)
    # 脚質: レース内で前に行きそうな順位
    d["pos1_rk"] = d.groupby("race_id")["pos1_r_m3"].rank(pct=True)
    return d


REL = [c + s for c in ["fig_m3", "fig_max5", "fig_p1", "lodds_m3", "fin_r_m3", "jockey_rate", "fig_same_surf"]
       for s in ("_rel", "_rk")] + ["pos1_rk"]


def encode_cats(d: pd.DataFrame, cats_map: dict | None = None):
    cats_map = cats_map or {}
    for c, src in [("venue_c", "venue"), ("surface_c", "surface"), ("course_c", "course"), ("direction_c", "direction")]:
        if c not in cats_map:
            cats_map[c] = {v: i for i, v in enumerate(sorted(d[src].dropna().unique()))}
        d[c] = d[src].map(cats_map[c]).astype("float")
    return d, cats_map
