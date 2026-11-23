import sys; sys.path.insert(0,'.')
from datetime import datetime, timedelta
from keirin.config import load_config
from keirin.db import DB
from keirin import pipeline
from keirin.cli import make_source
cfg=load_config('config.toml')
db=DB(cfg.path('db')); src=make_source(cfg)
d=datetime.now().strftime('%Y-%m-%d'); now=datetime.now()
rs=db.df("select * from races where date=? and has_result=0",(d,)).to_dict('records')
n=0
for r in rs:
    st=datetime.strptime(f"{d} {r['start_time']}","%Y-%m-%d %H:%M")
    if now-st<timedelta(minutes=15): continue
    try: n+=pipeline.ingest_result(db,src,r)
    except Exception as e: print('fail',r['race_id'],e)
print('results',n,'settled',pipeline.settle(db))
