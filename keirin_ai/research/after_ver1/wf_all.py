import sys, logging, time
sys.path.insert(0, '/home/user/uni/keirin_ai')
logging.basicConfig(level=logging.WARNING)
import pandas as pd
from keirin.config import load_config
from keirin.db import DB
from keirin import pipeline
from keirin.model import make_methods
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
FOLDS=[("2026-06-11","2026-07-10"),("2026-07-11","2026-08-10"),("2026-08-11","2026-09-10"),("2026-09-11","2026-10-10"),("2026-10-11","2026-11-20")]
cfg=load_config('/home/user/uni/keirin_ai/config.toml'); db=DB(f'{S}/sim.sqlite')
feats=pipeline.features_for(db,cfg); feats=feats[feats.finish.notna()]
names=pipeline.method_names(cfg); parts=[]; t0=time.time()
for a,b in FOLDS:
    train=feats[feats.date<a]; test=feats[(feats.date>=a)&(feats.date<=b)]
    for m in make_methods(names):
        m.fit(train); p=m.predict(test)["p_win"]
        parts.append(test[["race_id","date","car_no","finish","line_id","line_pos","line_size"]].assign(p=p.values,method=m.name))
    print(a,f"{time.time()-t0:.0f}s",flush=True)
pd.concat(parts).to_pickle(f'{S}/wf_all.pkl'); print("done",flush=True)
