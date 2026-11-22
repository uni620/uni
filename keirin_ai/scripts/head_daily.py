"""頭AI v1 ＋ 相手AI（展開型）v1 で今日のレースを予想して記録する（keirin/head.py）。
predictions_m に method='頭AI v1'（1着確率）、自信60%以上のレースは仮想購入（2車単3点・3連単9点）を bets に記録。
毎朝の処理（cloud_daily.sh）の最後に動かす。通信はしない。--dry は記録せず表示だけ。"""
import sys
from datetime import date, datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keirin.db import DB  # noqa: E402
from keirin.head import HEAD_NAME, predict  # noqa: E402
args = [a for a in sys.argv[1:] if not a.startswith('--')]
dry = '--dry' in sys.argv
day = args[0] if args else date.today().isoformat()
db = DB(str(ROOT / "data" / "keirin.sqlite"))
pr, picks = predict(db.con, day)
info = {r[0]: (r[1], r[2], r[3]) for r in db.con.execute("SELECT race_id, venue, race_no, start_time FROM races WHERE date=?", (day,))}
ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
n = 0
if not dry:
    for rid, g in pr.groupby("race_id"):
        db.save_method_preds(rid, HEAD_NAME, g.car_no.astype(int).tolist(), g.pr.tolist())
for p in sorted(picks, key=lambda p: info.get(p['race_id'], ('', 0, ''))[2] or ''):
    v, rn, st = info.get(p['race_id'], ('?', '?', '?'))
    print(f"{v}{rn}R {st} 頭{p['head']}（{p['conf']*100:.0f}%）主導権{p['b']} 2着候補{'-'.join(map(str, p['sec']))}")
    if dry:
        continue
    scratched = db.con.execute("SELECT 1 FROM entries WHERE race_id=? AND scratched=1", (p['race_id'],)).fetchone()
    if scratched:
        continue
    for strat, bets in p['bets'].items():
        for bt, combo in bets:
            cur = db.con.execute(
                """INSERT OR IGNORE INTO bets (mode, race_id, date, strategy, bet_type, combo, stake, prob, odds,
                       odds_final, ev, decided_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("live", p['race_id'], day, strat, bt, combo, 100, p['conf'], 0, 0, 0, ts))
            n += cur.rowcount
db.commit()
print(f"頭AI v1: {pr.race_id.nunique()} レースを予想、自信60%以上 {len(picks)} レース、仮想購入 {n} 点（{day}）")
