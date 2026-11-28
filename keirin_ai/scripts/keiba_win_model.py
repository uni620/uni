"""単勝予想モデルを学習・検証し、出馬表（data/keiba_today/*.html）の勝率を出す（個人の分析用）。

使い方: python scripts/keiba_win_model.py [出馬表のレースID ...]
検証: 2026年のレースを学習に使わずに当て、的中率・回収率を市場（最終オッズ）と比べる。
"""
import re
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keiba.features import (CATS, FEATURES, REL, add_relative, encode_cats, history_features, load,  # noqa: E402
                            prepare, rate_features)

TODAY_IDS = sys.argv[1:]
VENUE = {"05": "東京", "08": "京都", "06": "中山", "09": "阪神", "07": "中京", "01": "札幌", "02": "函館", "03": "福島",
         "04": "新潟", "10": "小倉"}


def parse_shutuba(rid: str) -> pd.DataFrame:
    h = (ROOT / "data" / "keiba_today" / f"{rid}.html").read_text()
    s = BeautifulSoup(h, "html.parser")
    rd = s.select_one(".RaceData01").get_text(" ", strip=True)
    m = re.search(r"(芝|ダ|障)(\d+)m\s*(右|左|直線)?", rd)
    g = re.search(r"(?:予報|馬場:)\s*(良|稍|重|不)", rd)
    rn = s.select_one(".RaceName")
    name = rn.get_text(strip=True)
    gi = re.search(r"Icon_GradeType(\d+)", str(rn.parent))
    name += {"1": "(GI)", "2": "(GII)", "3": "(GIII)"}.get(gi.group(1), "") if gi else ""
    d2 = s.select_one(".RaceData02").get_text(" ", strip=True) if s.select_one(".RaceData02") else ""
    rows = []
    for tr in s.select("tr.HorseList"):
        tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if len(tds) < 8 or not tds[1].isdigit():
            continue
        hrefs = [a.get("href", "") for a in tr.find_all("a")]
        hid = next((re.search(r"/horse/(\w+)", x).group(1) for x in hrefs if re.search(r"db\.netkeiba\.com/horse/\w+", x)), None)
        jid = next((re.search(r"/jockey/result/recent/(\w+)", x).group(1) for x in hrefs if "/jockey/" in x), None)
        tid = next((re.search(r"/trainer/result/recent/(\w+)", x).group(1) for x in hrefs if "/trainer/" in x), None)
        cancelled = "Cancel" in " ".join(tr.get("class", []))
        rows.append(dict(race_id=rid, umaban=int(tds[1]), waku=int(tds[0]), horse_name=tds[3], horse_id=hid,
                         sex=tds[4][:1], age=int(re.sub(r"\D", "", tds[4]) or 0), weight_carried=float(tds[5] or 0),
                         jockey=tds[6], jockey_id=jid, trainer_id=tid, cancelled=cancelled))
    df = pd.DataFrame(rows)
    df["venue"] = VENUE[rid[4:6]]
    df["surface"], df["distance"], df["direction"] = m.group(1), int(m.group(2)), m.group(3)
    df["going"] = {"稍": "稍重", "不": "不良"}.get(g.group(1), g.group(1)) if g else "良"
    df["race_name"] = name
    df["class_text"] = d2
    return df


def main():
    raw = load(str(ROOT / "data" / "keiba.sqlite"))
    today = pd.concat([parse_shutuba(r) for r in TODAY_IDS]) if TODAY_IDS else pd.DataFrame()
    if len(today):
        today = today[~today["cancelled"]].copy()
        today["date"] = "2026-11-29"
        today["finish_text"] = None
        raw = pd.concat([raw, today.drop(columns=["cancelled", "horse_name", "jockey"])], ignore_index=True)
    d = prepare(raw)
    d["is_today"] = d["race_id"].isin(TODAY_IDS)
    d.loc[d["is_today"], "n"] = d[d["is_today"]].groupby("race_id")["umaban"].transform("count")
    d = history_features(d)
    d = rate_features(d, "jockey_id", "jockey_rate")
    d = rate_features(d, "trainer_id", "trainer_rate")
    d = add_relative(d)
    d, _ = encode_cats(d)
    feats = FEATURES + REL
    hist = d[~d["is_today"] & (d["date"] >= "2020-04-01") & (d["surface"] != "障")]
    train = hist[hist["date"] < "2026-01-01"]
    test = hist[hist["date"] >= "2026-01-01"]
    params = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=1)
    tr_ds = lgb.Dataset(train[feats], train["win"], categorical_feature=CATS)
    te_ds = lgb.Dataset(test[feats], test["win"], categorical_feature=CATS)
    m = lgb.train(params, tr_ds, 2000, valid_sets=[te_ds], callbacks=[lgb.early_stopping(100, verbose=False)])
    best = m.best_iteration
    test = test.copy()
    test["p"] = m.predict(test[feats], num_iteration=best)
    test["p"] = test["p"] / test.groupby("race_id")["p"].transform("sum")
    report(test)
    test.to_csv(ROOT / "data" / "keiba_today" / "test_pred.csv", index=False)
    # 本番: 全期間で学習し直す
    full = lgb.train(params, lgb.Dataset(hist[feats], hist["win"], categorical_feature=CATS), int(best * 1.1))
    imp = pd.Series(full.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("\n重要な特徴（上位15）:\n", (imp / imp.sum()).head(15).round(3).to_string())
    if len(today):
        t = d[d["is_today"]].copy()
        t["p"] = full.predict(t[feats])
        t["p"] = t["p"] / t.groupby("race_id")["p"].transform("sum")
        t = t.merge(today[["race_id", "umaban", "horse_name", "jockey"]], on=["race_id", "umaban"])
        t.to_csv(ROOT / "data" / "keiba_today" / "pred.csv", index=False)
        for rid, g in t.groupby("race_id"):
            g = g.sort_values("p", ascending=False)
            print(f"\n=== {rid} {g['venue'].iloc[0]}{int(rid[-2:])}R {g['race_name'].iloc[0]} {g['surface'].iloc[0]}{g['distance'].iloc[0]}m")
            for _, x in g.iterrows():
                print(f"  {int(x.umaban):2d} {x.horse_name:<12} 勝率{x.p*100:5.1f}%  妥当オッズ{1/x.p:6.1f}倍  "
                      f"過去走{int(x.n_runs)} 指数3走{x.fig_m3:5.2f} 前走着{x.fin_r_p1 if pd.notna(x.fin_r_p1) else float('nan'):.2f} 騎手勝率{x.jockey_rate*100:4.1f}%")


def report(t: pd.DataFrame):
    t = t[t["win_odds"].notna()].copy()
    t["q"] = (1 / t["win_odds"])
    t["q"] = t["q"] / t.groupby("race_id")["q"].transform("sum")
    eps = 1e-6
    ll = lambda p: -np.mean(t["win"] * np.log(p + eps) + (1 - t["win"]) * np.log(1 - p + eps))  # noqa: E731
    print(f"検証（2026年 {t.race_id.nunique()}レース）: logloss モデル {ll(t.p):.4f} / 市場 {ll(t.q):.4f}")
    top = t.loc[t.groupby("race_id")["p"].idxmax()]
    fav = t.loc[t.groupby("race_id")["q"].idxmax()]
    print(f"  モデル1位の単勝: 的中 {top.win.mean()*100:.1f}%  回収 {(top.win*top.win_odds).mean()*100:.0f}%")
    print(f"  1番人気の単勝  : 的中 {fav.win.mean()*100:.1f}%  回収 {(fav.win*fav.win_odds).mean()*100:.0f}%")
    for th in [1.0, 1.2, 1.5, 2.0]:
        b = t[t.p * t.win_odds >= th]
        print(f"  期待値{th:.1f}以上を全部買う: {len(b)}点 的中{b.win.mean()*100:.1f}% 回収{(b.win*b.win_odds).mean()*100:.0f}%")
    # 確率の当たり具合（予想勝率の帯ごとの実際の勝率）
    t["bin"] = pd.cut(t.p, [0, .02, .05, .1, .2, .3, .5, 1])
    print(t.groupby("bin", observed=True).agg(n=("win", "size"), 予想=("p", "mean"), 実際=("win", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
