"""ガールズのレースの出走表と結果・払戻を埋める（2026-11-22 からガールズも記録する）。
出走表は保存済みのページがあればそれを使う。並び予想は取らない（全員単騎）。間隔は config の request_interval_sec。"""
import sys, logging
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keirin.config import load_config  # noqa: E402
from keirin.db import DB  # noqa: E402
from keirin.sources import make_source  # noqa: E402
from keirin import pipeline  # noqa: E402
logging.basicConfig(level=logging.WARNING)
cfg = load_config(str(ROOT / (sys.argv[1] if len(sys.argv) > 1 else "config.toml"))); db = DB(cfg.path("db")); src = make_source(cfg)
races = db.df("""SELECT * FROM races WHERE race_class LIKE '%ガールズ%' AND has_result=0 AND date < date('now','localtime')
                 ORDER BY date DESC""").to_dict("records")
print("対象", len(races), flush=True)
ok = 0
for i, r in enumerate(races):
    try:
        if not db.con.execute("SELECT 1 FROM entries WHERE race_id=? LIMIT 1", (r["race_id"],)).fetchone():
            card = src._card(r["race_id"])
            if card and card.entries:
                db.upsert_entries(card.entries); db.upsert_entry_info(card.entries)
                db.insert_honshi_bets(r["race_id"], card.race.get("honshi_bets") or [])
        ok += pipeline.ingest_result(db, src, r)
        db.commit()
    except Exception as e:  # noqa: BLE001
        print("失敗", r["race_id"], str(e)[:80], flush=True)
    if (i + 1) % 200 == 0:
        print(f"{i+1}/{len(races)} 結果{ok} 直近 {r['date']}", flush=True)
print(f"--- 終了 結果{ok}/{len(races)}", flush=True)
