"""出走表 → AIに渡す特徴量。

ポイント
- 競輪は「同じレースの中での相対的な強さ」で勝負が決まるので、多くの特徴量を
  「レース平均との差」にしている（例: 競走得点がレース平均より何点高いか）。
- 直近成績は「そのレースより前のレース」だけから計算する（未来の情報を使わない）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLASS_NUM = {"SS": 6, "S1": 5, "S2": 4, "A1": 3, "A2": 2, "A3": 1, "L1": 2}

FEATURES = [
    "score_z", "score_rank", "class_diff",
    "is_nige", "is_ryo",
    "is_line_head", "is_bante", "is_third_plus", "is_tanki",
    "line_strength", "line_size", "head_nige",
    "home", "age_diff",
    "form_win", "form_top2", "form_top3", "form_n",
]

_PRIOR = {"win": 1 / 8, "top2": 2 / 8, "top3": 3 / 8}
_SHRINK_K = 5.0


def _form_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.sort_values(["date", "venue", "race_no", "car_no"]).copy()
    valid = df["finish"].notna() & (df["scratched"] == 0)
    fin = df["finish"].where(valid)
    df["_n"] = valid.astype(float)
    df["_w"] = (fin == 1).astype(float).where(valid, 0.0)
    df["_t2"] = (fin <= 2).astype(float).where(valid, 0.0)
    df["_t3"] = (fin <= 3).astype(float).where(valid, 0.0)
    g = df.groupby("rider_id", sort=False)
    roll = {}
    for c in ["_n", "_w", "_t2", "_t3"]:
        roll[c] = g[c].transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum()).fillna(0.0)
    n = roll["_n"]
    df["form_win_raw"] = (roll["_w"] + _PRIOR["win"] * _SHRINK_K) / (n + _SHRINK_K)
    df["form_top2_raw"] = (roll["_t2"] + _PRIOR["top2"] * _SHRINK_K) / (n + _SHRINK_K)
    df["form_top3_raw"] = (roll["_t3"] + _PRIOR["top3"] * _SHRINK_K) / (n + _SHRINK_K)
    df["form_n"] = np.log1p(n)
    return df.drop(columns=["_n", "_w", "_t2", "_t3"])


def build_features(df: pd.DataFrame, form_window: int = 10) -> pd.DataFrame:
    """entries_with_race() の結果を受け取り、欠車を除いた行に特徴量列を追加して返す。"""
    if df.empty:
        return df.assign(**{f: [] for f in FEATURES})
    df = df.copy()
    for c in ["score", "age", "line_pos", "line_size", "home", "scratched"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["scratched"] = df["scratched"].fillna(0).astype(int)
    df["home"] = df["home"].fillna(0)
    df = _form_features(df, form_window)
    df = df[df["scratched"] == 0].copy()

    g = df.groupby("race_id")
    df["score"] = df["score"].fillna(g["score"].transform("mean")).fillna(0)
    mean = g["score"].transform("mean")
    std = g["score"].transform("std").fillna(0)
    df["score_z"] = (df["score"] - mean) / (std + 1.0)
    df["score_rank"] = g["score"].rank(ascending=False, pct=True)

    df["class_num"] = df["class_rank"].map(CLASS_NUM).astype(float)
    df["class_num"] = df["class_num"].fillna(g["class_num"].transform("mean")).fillna(3)
    df["class_diff"] = df["class_num"] - g["class_num"].transform("mean")

    style = df["style"].fillna("")
    df["is_nige"] = style.str.contains("逃").astype(float)
    df["is_ryo"] = style.str.contains("両").astype(float)

    df["line_size"] = df["line_size"].fillna(1)
    df["line_pos"] = df["line_pos"].fillna(1)
    df["is_tanki"] = (df["line_size"] <= 1).astype(float)
    df["is_line_head"] = ((df["line_pos"] == 1) & (df["line_size"] >= 2)).astype(float)
    df["is_bante"] = (df["line_pos"] == 2).astype(float)
    df["is_third_plus"] = (df["line_pos"] >= 3).astype(float)
    line_key = df["race_id"] + "_" + df["line_id"].fillna(-df["car_no"]).astype(int).astype(str)
    line_mean = df.groupby(line_key)["score"].transform("mean")
    df["line_strength"] = (line_mean - mean) / (std + 1.0)
    df["head_nige"] = df["is_line_head"] * df["is_nige"]

    df["age"] = df["age"].fillna(g["age"].transform("mean")).fillna(30)
    df["age_diff"] = (df["age"] - g["age"].transform("mean")) / 10.0

    for k in ["win", "top2", "top3"]:
        raw = df[f"form_{k}_raw"]
        df[f"form_{k}"] = raw - df.groupby("race_id")[f"form_{k}_raw"].transform("mean")
    df["form_n"] = df["form_n"] - g["form_n"].transform("mean")

    df[FEATURES] = df[FEATURES].astype(float).fillna(0.0)
    return df
