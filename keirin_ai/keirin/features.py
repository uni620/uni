"""出走表 → AIに渡す列（設計書「データ項目一覧」どおり）。

1行 = 1レースの1選手。すべて「予想するレースより前」の情報だけで計算する（リーク防止）。
競走得点・脚質・級班は使わない（設計書「使わないデータ」）。比較用に score_diff だけ別に作る。
ガールズケイリンは対象外。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .race_grade import is_semifinal, kaku

# AIに渡す列（内部名）。日本語名は FEATURE_JA。
FEATURES = [
    # 1 位置の適性
    "jiriki", "kimarite_n", "pos_type", "pos_fit",
    # 2 強いライン
    "line_gap", "other_senko_n",
    # 3 強さ
    "recent_finish", "is_current_meet", "days_since_prev",
    # 4 調子
    "win_3m", "top2_3m", "top3_3m", "n_3m", "trend",
    # 5 格
    "kaku_now", "kaku_usual", "kaku_diff", "kaku_member_diff", "race_class_code", "semi_finish",
]
# 保留中の列: 計算はするが AI には渡さない。
# 2026-11-19 に 925 レースで比べたところ 1着率はほぼ変わらなかった（LightGBM 43.9%→43.9%）。
# 過去データが少なく、同じ位置の過去出走が平均約7回しかないため。データが増えたら FEATURES に足して比べ直す。
POS_RECORD_FEATURES = ["pos_n", "pos_win", "pos_top3", "pos_top3_diff"]
# バンクの適性（保留中・比べてから決める）
BANK_FEATURES = ["bank_cls", "straight_len", "bank_n", "bank_top3", "bank_top3_diff",
                 "venue_n", "venue_top3", "venue_top3_diff"]
CATEGORICAL = ["pos_type", "race_class_code"]
SCORE_FEATURE = "score_diff"   # 「機械学習＋競走得点」の比較用

FEATURE_JA = {
    "jiriki": "自力度", "kimarite_n": "決まり手の合計回数", "pos_type": "今回の位置",
    "pos_fit": "位置の適性", "pos_n": "同じ位置の過去出走数", "pos_win": "同じ位置での勝率",
    "pos_top3": "同じ位置での複勝率", "pos_top3_diff": "同じ位置での複勝率−全体の複勝率",
    "bank_cls": "バンクの周長（333/400/500）", "straight_len": "見なし直線の長さ",
    "bank_n": "同じ周長の過去出走数", "bank_top3": "同じ周長での複勝率", "bank_top3_diff": "同じ周長での複勝率−全体",
    "venue_n": "同じ競輪場の過去出走数", "venue_top3": "同じ競輪場での複勝率", "venue_top3_diff": "同じ競輪場での複勝率−全体",
"line_gap": "ラインの力量差", "other_senko_n": "ほかのラインの先行選手の数",
    "recent_finish": "直近の大会の着順（平均）", "is_current_meet": "どちらの大会か（今大会=1）",
    "days_since_prev": "前大会からの日数", "win_3m": "3か月の勝率", "top2_3m": "3か月の連対率",
    "top3_3m": "3か月の複勝率", "n_3m": "3か月の出走数", "trend": "調子の上向き度",
    "kaku_now": "今回の格", "kaku_usual": "普段の格", "kaku_diff": "格の差（＋=格下のレース）",
    "kaku_member_diff": "メンバー内の格の差", "race_class_code": "レースのクラス",
    "semi_finish": "準決勝の着順", "score_diff": "競走得点（レース平均との差）",
}

POS_TYPES = {"先頭": 0, "番手": 1, "3番手": 2, "単騎": 3}
POS_PRIOR = 3.0   # 位置別の成績を、全体の成績に何走分寄せるか（回数が少ない位置のブレを抑える）
CLASS_CODES = {"A級チャレンジ": 0, "A級": 1, "S級": 2}
JIRIKI_KIMARITE = {"逃げ", "捲り"}
OTHER_KIMARITE = {"差し", "マーク"}


def _norm_kimarite(k) -> str | None:
    if k is None or (isinstance(k, float) and np.isnan(k)):
        return None
    k = str(k).strip()
    # netkeirin の結果表は 逃/捲/差/ク（ク = マーク の略と判断）で表示される
    for full, keys in {"逃げ": ("逃",), "捲り": ("捲", "まくり"), "差し": ("差",), "マーク": ("マ", "ク")}.items():
        if any(x in k for x in keys):
            return full
    return None


def _class_code(c) -> float:
    if c is None or (isinstance(c, float) and np.isnan(c)):
        return np.nan
    c = str(c)
    if "チャレンジ" in c:
        return 0
    if c.startswith("S"):
        return 2
    if c.startswith("A"):
        return 1
    return np.nan


def is_girls(c) -> bool:
    return isinstance(c, str) and ("ガールズ" in c or c.startswith("L"))


def assign_meets(df: pd.DataFrame) -> pd.Series:
    """同じ場で日付が連続している開催を1つの大会とみなす（間が2日以上あいたら別大会）。"""
    days = df[["venue", "date"]].drop_duplicates().sort_values(["venue", "date"])
    d = pd.to_datetime(days["date"])
    new = (days["venue"] != days["venue"].shift()) | ((d - d.shift()).dt.days > 1)
    days["meet_id"] = days["venue"] + "_" + days["date"].where(new).ffill()
    return df.merge(days, on=["venue", "date"], how="left")["meet_id"].to_numpy()


class _Record:
    """「今回と同じ条件（位置・バンクなど）」で過去に走ったときの成績。自分より前のレースだけを数える。
    回数が少ないとブレるので、その選手の全体の成績に POS_PRIOR 走分だけ寄せて補正する。"""

    def __init__(self, df: pd.DataFrame, fin: pd.Series, valid: pd.Series):
        self.rider = df["rider_id"]
        self.ok = valid.astype(float)
        self.w = (fin == 1).astype(float).where(valid, 0.0)
        self.t3 = (fin <= 3).astype(float).where(valid, 0.0)
        g = lambda s: s.groupby(self.rider).cumsum() - s
        self.an = g(self.ok)
        self.all_win = (g(self.w) / self.an).where(self.an > 0)
        self.all_t3 = (g(self.t3) / self.an).where(self.an > 0)

    def by(self, key: pd.Series):
        """(同じ条件の過去出走数, 補正した勝率, 補正した複勝率, 複勝率−全体の複勝率) を返す。"""
        k = key.astype(object).where(key.notna(), "_na")
        c = lambda s: s.groupby([self.rider, k], sort=False).cumsum() - s
        n, w, t3 = c(self.ok), c(self.w), c(self.t3)
        has = (self.an > 0) & key.notna()
        win = ((w + POS_PRIOR * self.all_win) / (n + POS_PRIOR)).where(has)
        top3 = ((t3 + POS_PRIOR * self.all_t3) / (n + POS_PRIOR)).where(has)
        return n.where(key.notna()), win, top3, top3 - self.all_t3


def build_features(raw: pd.DataFrame, jiriki_threshold: float = 0.6, form_days: int = 90,
                   trend_races: int = 4, kaku_overrides: dict | None = None) -> pd.DataFrame:
    if raw.empty:
        return raw.assign(**{f: [] for f in FEATURES + [SCORE_FEATURE]})
    df = raw.copy()
    df = df[~df["race_class"].apply(is_girls)].copy()
    for c in ["finish", "line_pos", "line_size", "score", "scratched"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["scratched"] = df["scratched"].fillna(0).astype(int)
    df["dt"] = pd.to_datetime(df["date"])
    df = df.sort_values(["dt", "venue", "race_no", "car_no"]).reset_index(drop=True)
    df["meet_id"] = assign_meets(df)
    valid = df["finish"].notna() & (df["scratched"] == 0)
    fin = df["finish"].where(valid)

    g = df.groupby("rider_id", sort=False)

    # ---------- 1 位置の適性 ----------
    km = df["kimarite"].map(_norm_kimarite)
    df["_kj"] = km.isin(JIRIKI_KIMARITE).astype(float)
    df["_ko"] = km.isin(OTHER_KIMARITE).astype(float)
    cj = g["_kj"].cumsum() - df["_kj"]          # 自分より前のレースだけの累計
    co = g["_ko"].cumsum() - df["_ko"]
    # 出走表に決まり手の回数が載っていれば（レース前に公開される情報）、そちらを優先する
    km_cols = ["km_nige", "km_makuri", "km_sashi", "km_mark"]
    for c in km_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    has_km = df[km_cols].notna().all(axis=1)
    cj = cj.where(~has_km, df["km_nige"] + df["km_makuri"])
    co = co.where(~has_km, df["km_sashi"] + df["km_mark"])
    df["kimarite_n"] = cj + co
    df["jiriki"] = (cj / df["kimarite_n"]).where(df["kimarite_n"] > 0)
    size = df["line_size"].fillna(1)
    pos = df["line_pos"].fillna(1)
    df["pos_type"] = np.select([size <= 1, pos == 1, pos == 2], [3, 0, 1], default=2).astype(float)
    df["pos_fit"] = np.select([df["pos_type"] == 0, df["pos_type"] == 1], [df["jiriki"], 1 - df["jiriki"]],
                              default=np.nan)

    # 位置ごとの実績: 過去に「今回と同じ位置」を走ったときの成績（勝率・複勝率を位置別に分けて見る）
    rec = _Record(df, fin, valid)
    df["pos_n"], df["pos_win"], df["pos_top3"], df["pos_top3_diff"] = rec.by(df["pos_type"])

    # バンクの適性: 同じ周長（333/400/500m）・同じ競輪場での過去の成績
    bank = pd.to_numeric(df["bank_len"], errors="coerce") if "bank_len" in df else pd.Series(np.nan, index=df.index)
    df["bank_cls"] = np.select([bank <= 340, bank < 450, bank >= 450], [0, 1, 2], default=np.nan)
    df["straight_len"] = pd.to_numeric(df["straight_len"], errors="coerce") if "straight_len" in df else np.nan
    df["bank_n"], _, df["bank_top3"], df["bank_top3_diff"] = rec.by(df["bank_cls"])
    df["venue_n"], _, df["venue_top3"], df["venue_top3_diff"] = rec.by(df["venue"])

    # ---------- 3 強さ（今大会・前大会の着順） ----------
    df["_fin"] = fin
    df["_ok"] = valid.astype(float)
    df["_finz"] = fin.fillna(0)
    gm = df.groupby(["rider_id", "meet_id"], sort=False)
    s_before = gm["_finz"].cumsum() - df["_finz"]
    n_before = gm["_ok"].cumsum() - df["_ok"]
    cur = (s_before / n_before).where(n_before > 0)
    meet_tbl = df.groupby(["rider_id", "meet_id"], sort=False).agg(
        m_sum=("_finz", "sum"), m_n=("_ok", "sum"), m_start=("dt", "min"), m_end=("dt", "max")).reset_index()
    meet_tbl = meet_tbl.sort_values(["rider_id", "m_start"])
    mg = meet_tbl.groupby("rider_id", sort=False)
    meet_tbl["prev_avg"] = (meet_tbl["m_sum"] / meet_tbl["m_n"].replace(0, np.nan)).groupby(meet_tbl["rider_id"]).shift(1)
    meet_tbl["prev_end"] = mg["m_end"].shift(1)
    df = df.merge(meet_tbl[["rider_id", "meet_id", "prev_avg", "prev_end"]], on=["rider_id", "meet_id"], how="left")
    df["recent_finish"] = cur.fillna(df["prev_avg"])
    df["is_current_meet"] = cur.notna().astype(float)
    df["days_since_prev"] = (df["dt"] - df["prev_end"]).dt.days

    # ---------- 4 調子（直近3か月・直近4走） ----------
    df["_w"] = (fin == 1).astype(float).where(valid, 0.0)
    df["_t2"] = (fin <= 2).astype(float).where(valid, 0.0)
    df["_t3"] = (fin <= 3).astype(float).where(valid, 0.0)
    roll_cols = ["_ok", "_w", "_t2", "_t3"]
    r = (df.set_index("dt").groupby("rider_id", sort=True)[roll_cols]
           .rolling(f"{form_days}D", closed="left").sum().reset_index(level=0, drop=True))
    # rolling の結果は rider ごとに並ぶので、元の順番に戻す
    order = df.sort_values(["rider_id", "dt"], kind="stable").index
    r.index = order
    r = r.sort_index()
    n3 = r["_ok"]
    df["n_3m"] = n3
    df["win_3m"] = (r["_w"] / n3).where(n3 > 0)
    df["top2_3m"] = (r["_t2"] / n3).where(n3 > 0)
    df["top3_3m"] = (r["_t3"] / n3).where(n3 > 0)
    g = df.groupby("rider_id", sort=False)
    last_t3 = g["_t3"].transform(lambda s: s.shift(1).rolling(trend_races, min_periods=trend_races).sum())
    last_n = g["_ok"].transform(lambda s: s.shift(1).rolling(trend_races, min_periods=trend_races).sum())
    df["trend"] = (last_t3 / last_n).where(last_n > 0) - df["top3_3m"]

    # ---------- 5 格 ----------
    df["kaku_now"] = [kaku(n, gr, kaku_overrides) for n, gr in zip(df["race_name"], df["grade"])]
    df["kaku_now"] = df["kaku_now"].astype(float)
    df["_kk"] = df["kaku_now"].fillna(0)
    df["_kn"] = df["kaku_now"].notna().astype(float)
    ks = g["_kk"].cumsum() - df["_kk"]
    kn = g["_kn"].cumsum() - df["_kn"]
    df["kaku_usual"] = (ks / kn).where(kn > 0)
    df["kaku_diff"] = df["kaku_usual"] - df["kaku_now"]
    df["race_class_code"] = df["race_class"].map(_class_code)
    semi = df["race_name"].map(is_semifinal) & valid
    df["_semi"] = fin.where(semi)
    df["semi_finish"] = (df.groupby(["rider_id", "meet_id"], sort=False)["_semi"]
                           .transform(lambda s: s.shift(1).ffill()))

    # ---------- ここから先はレース内での比較（欠車は除く） ----------
    df = df[df["scratched"] == 0].copy()
    rg = df.groupby("race_id")
    tot = rg["kaku_usual"].transform("sum")
    cnt = rg["kaku_usual"].transform("count")
    own = df["kaku_usual"]
    others_mean = (tot - own.fillna(0)) / (cnt - own.notna().astype(int)).replace(0, np.nan)
    df["kaku_member_diff"] = own - others_mean

    # ---------- 2 強いライン ----------
    df["_strength"] = -df["recent_finish"]
    line_key = df["race_id"] + "_" + df["line_id"].fillna(-df["car_no"]).astype(int).astype(str)
    df["_line"] = line_key
    df["_senko"] = (df["jiriki"] >= jiriki_threshold).astype(int)
    race_senko = df.groupby("race_id")["_senko"].transform("sum")
    line_senko = df.groupby("_line")["_senko"].transform("sum")
    df["other_senko_n"] = (race_senko - line_senko).astype(float)

    top2 = df[(df["line_size"].fillna(1) >= 2) & (df["line_pos"] <= 2)]
    weaker = top2.groupby("_line")["_strength"].min()           # NaN があれば NaN（下の min は skipna なので調整）
    has_nan = top2.groupby("_line")["_strength"].apply(lambda s: s.isna().any())
    weaker[has_nan] = np.nan
    gaps = {}
    for rid, rr in df.groupby("race_id"):
        for ln in rr["_line"].unique():
            if ln not in weaker.index:
                continue
            others = rr.loc[rr["_line"] != ln, "_strength"]
            gaps[ln] = weaker[ln] - others.max() if len(others) and others.notna().any() else np.nan
    df["line_gap"] = np.where((df["line_size"].fillna(1) >= 2) & (df["line_pos"] <= 2),
                              df["_line"].map(gaps), np.nan)

    # ---------- 比較用: 競走得点 ----------
    df["score_diff"] = df["score"] - df.groupby("race_id")["score"].transform("mean")

    # ---------- 答えの列 ----------
    f = df["finish"]
    df["y_win"] = (f == 1).astype(float).where(f.notna())
    df["y_top3"] = (f <= 3).astype(float).where(f.notna())
    pair_rows = (df["line_size"].fillna(1) >= 2) & (df["line_pos"] <= 2)
    both = df[pair_rows].groupby("_line")["finish"].apply(lambda s: float((s <= 2).sum() == 2) if s.notna().all() else np.nan)
    df["y_pair"] = np.where(pair_rows, df["_line"].map(both), np.nan)

    drop = [c for c in df.columns if c.startswith("_") and c != "_line"] + ["prev_avg", "prev_end"]
    return df.drop(columns=drop).reset_index(drop=True)
