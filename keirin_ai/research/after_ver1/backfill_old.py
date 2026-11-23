"""保存済みの netkeirin のページ（data/cache）から、選手コメント・本紙の印と買い目・上り/着差/SB・配当の人気を DB に入れる。
新しい通信はしない。"""
import hashlib, sys
from pathlib import Path
ROOT = Path('/home/user/uni/keirin_ai')
sys.path.insert(0, str(ROOT))
from keirin.db import DB  # noqa: E402
from keirin.sources.netkeirin import BASE, parse_entry, parse_result  # noqa: E402
db = DB(str(ROOT / "data" / "keirin.sqlite")); C = ROOT / "data" / "cache"
page = lambda u: (C / (hashlib.sha1(u.encode()).hexdigest() + ".html"))
rids = [r[0] for r in db.con.execute("SELECT race_id FROM races WHERE date < '2026-06-11' ORDER BY date")]
ne = nr = 0
for i, rid in enumerate(rids):
    p = page(f"{BASE}/race/entry/?race_id={rid}")
    if p.exists():
        race, rows = parse_entry(p.read_text(encoding="utf-8"), rid)
        db.upsert_entry_info(rows); db.insert_honshi_bets(rid, race.get("honshi_bets") or []); ne += 1
    p = page(f"{BASE}/race/result/?race_id={rid}")
    if p.exists():
        res = parse_result(p.read_text(encoding="utf-8"), rid)
        if res:
            for bt, n in res.payout_ninki.items():
                db.upsert_payout_ninki(rid, bt, n)
            for bt, pz in res.payouts.items():
                db.upsert_payouts(rid, bt, pz)
            db.upsert_result_info(rid, res.rider_info); nr += 1
    db.commit()                      # 他の記録処理を待たせないよう、1レースごとに書き込む
    if (i + 1) % 1000 == 0:
        print(f"{i+1}/{len(rids)} 出走表{ne} 結果{nr}", flush=True)
db.commit(); print(f"--- 終了 出走表{ne} 結果{nr}", flush=True)
