"""ver1d（オッズのずれ・ワイドなし・10倍以下・EV1.0以上の1点）を、100%超えの候補の切り口ごとに集計する（締切8分前のオッズで記録したもの）。
候補（2026-11-23、8〜11月の確定オッズで100%を超えた切り口）: A級チャレンジ EV1.0/1.1/1.2、F2 EV1.1、A級 EV1.1/1.2、S級 EV1.0/1.2、
時間帯（モーニング・デイ・ナイター・ミッドナイト）EV1.1、全体 EV1.1/1.2。G3以上は悪かったので比較用に出す。"""
import sqlite3, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
c = sqlite3.connect(ROOT / "data" / "keirin.sqlite")
b = pd.read_sql("""SELECT b.race_id, b.date, b.ev, b.odds, b.status, b.payout, b.stake, r.race_class, r.grade,
                   (SELECT MIN(start_time) FROM races r2 WHERE r2.venue=r.venue AND r2.date=r.date) AS first
                   FROM bets b JOIN races r USING(race_id) WHERE b.mode='live' AND b.strategy LIKE 'ver1d%'""", c)
if b.empty:
    print("ver1d の記録はまだありません"); sys.exit()
b["級"] = b.race_class.astype(str).map(lambda s: "A級チャレンジ" if "チャレンジ" in s else ("S級" if "S級" in s else "A級"))
g = b.grade.fillna("").str.upper()
b["グレード"] = ["G3以上" if "G" in x else ("F1" if x == "F1" else "F2") for x in g]
h = b["first"].fillna("12:00").str[:2].astype(int) + b["first"].fillna("12:00").str[3:5].astype(int) / 60
b["時間帯"] = pd.cut(h, [0, 9.5, 12, 17, 24], labels=["モーニング", "デイ", "ナイター", "ミッドナイト"]).astype(str)
s = b[b.status != "open"]
rows = []
for lab, m in [("全体", s.ev >= 0)] + [(f"級={k}", s.級 == k) for k in ("A級チャレンジ", "A級", "S級")] + \
              [(f"グレード={k}", s.グレード == k) for k in ("F2", "F1", "G3以上")] + [(f"時間帯={k}", s.時間帯 == k) for k in ("モーニング", "デイ", "ナイター", "ミッドナイト")]:
    for t in (1.0, 1.1, 1.2):
        x = s[m & (s.ev >= t)]
        if len(x):
            rows.append((lab, t, len(x), int((x.payout > 0).sum()), round(x.payout.sum() / x.stake.sum() * 100, 1)))
print(pd.DataFrame(rows, columns=["切り口", "EV以上", "点", "的中", "回収率%"]).to_string(index=False))
print(f"未精算 {int((b.status=='open').sum())} 点")
