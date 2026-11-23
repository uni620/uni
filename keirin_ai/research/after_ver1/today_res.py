import sys, logging
from datetime import datetime, timedelta
sys.path.insert(0, "/home/user/uni/keirin_ai")
from keirin.config import Config
from keirin.db import DB
from keirin.sources import make_source
from keirin.pipeline import ingest_result, settle
from keirin.config import load_config; cfg = load_config("/home/user/uni/keirin_ai/config.toml")
db = DB(cfg.path("db")); src = make_source(cfg)
today = datetime.now().strftime("%Y-%m-%d"); now = datetime.now().strftime("%H:%M")
lim = (datetime.now() - timedelta(minutes=8)).strftime("%H:%M")
races = db.df("""SELECT * FROM races WHERE date=? AND has_result=0 AND start_time<=?
                 AND race_id IN (SELECT race_id FROM bets WHERE mode='live')""", (today, lim)).to_dict("records")
print("取り込み対象", len(races))
ok = 0
for r in races:
    try:
        ok += ingest_result(db, src, r)
    except Exception as e:
        print("fail", r["race_id"], e)
print("取り込み", ok, "精算", settle(db))
