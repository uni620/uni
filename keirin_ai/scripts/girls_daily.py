"""ガールズ頭AI v1 で今日のガールズを予想して記録する（keirin/girls_head.py）。
predictions_m に method='ガールズ頭AI v1'（1着確率）、頭の確率80%以上のレースは仮想購入（ワイド3点・2車単3点）を bets に記録。
毎朝の処理（cloud_daily.sh）の最後に動かす。通信はしない。--dry は記録せず表示だけ。"""
import sys
from datetime import date, datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keirin.db import DB  # noqa: E402
from keirin.girls_head import GIRLS_HEAD, predict  # noqa: E402
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
        db.save_method_preds(rid, GIRLS_HEAD, g.car_no.astype(int).tolist(), g.pr.tolist())
for p in sorted(picks, key=lambda p: info.get(p['race_id'], ('', 0, ''))[2] or ''):
    v, rn, st = info.get(p['race_id'], ('?', '?', '?'))
    print(f"ガールズ {v}{rn}R {st} 頭{p['head']}（{p['conf']*100:.0f}%）相手{'-'.join(map(str, p['sec']))}")
    if dry or db.con.execute("SELECT 1 FROM entries WHERE race_id=? AND scratched=1", (p['race_id'],)).fetchone():
        continue
    for strat, bets in p['bets'].items():
        for bt, combo, prob in bets:
            n += db.con.execute("""INSERT OR IGNORE INTO bets (mode, race_id, date, strategy, bet_type, combo, stake, prob, odds,
                   odds_final, ev, decided_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                   ("live", p['race_id'], day, strat, bt, combo, 100, prob, 0, 0, 0, ts)).rowcount
db.commit()
print(f"ガールズ頭AI v1: {pr.race_id.nunique()} レースを予想、確率80%以上 {len(picks)} レース、仮想購入 {n} 点（{day}）")
