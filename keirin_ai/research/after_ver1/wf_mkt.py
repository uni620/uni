import sys, logging, time
sys.path.insert(0, '/home/user/uni/keirin_ai')
logging.basicConfig(level=logging.WARNING)
import pandas as pd, numpy as np
from keirin.config import load_config
from keirin.db import DB
from keirin import pipeline
from keirin.model import LGBModel, evaluate
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
FOLDS = [("2026-09-09","2026-09-22"),("2026-09-23","2026-10-06"),("2026-10-07","2026-10-20"),("2026-10-21","2026-11-03"),("2026-11-04","2026-11-19")]
cfg = load_config(); db = DB(f'{S}/wf.sqlite')
feats = pipeline.features_for(db, cfg); feats = feats[feats.finish.notna()].copy()
pop = pd.read_pickle(f'{S}/popular.pkl').set_index('race_id')
f = feats.race_id.map(pop.f1); s2 = feats.race_id.map(pop.f2); s3 = feats.race_id.map(pop.f3)
feats["mkt_is1"] = (feats.car_no == f).astype(float); feats["mkt_is2"] = (feats.car_no == s2).astype(float); feats["mkt_is3"] = (feats.car_no == s3).astype(float)
sup1 = feats.race_id.map(pop.sup1)
feats["mkt_sup"] = np.where(feats.mkt_is1 == 1, sup1, np.where(feats.mkt_is2 == 1, feats.race_id.map(pop.sup2), np.where(feats.mkt_is3 == 1, feats.race_id.map(pop.sup3), np.nan)))
feats["mkt_fav_sup1"] = sup1; feats["mkt_fav_odds"] = feats.race_id.map(pop.fav_odds)
MKT = ["mkt_is1", "mkt_is2", "mkt_is3", "mkt_sup", "mkt_fav_sup1", "mkt_fav_odds"]
parts = []; ev = []
for a, b in FOLDS:
    train = feats[feats.date < a]; test = feats[(feats.date >= a) & (feats.date <= b)]
    for name, extra in [("競走得点", []), ("競走得点+人気", MKT)]:
        m = LGBModel(True); m.features = m.features + extra; m.name = "LGB+" + name
        m.fit(train); p = m.predict(test)
        ev.append({"fold": a, "方法": m.name, **evaluate(test, p)})
        parts.append(test[["race_id","date","car_no","finish","race_class","grade"]].assign(p=p["p_win"].values, method=m.name, fold=a))
pd.concat(parts).to_pickle(f'{S}/wf_mkt.pkl')
e = pd.DataFrame(ev); pd.set_option('display.width', 200)
print(e.groupby("方法")[["1位予想の1着率", "予想のズレ(小さいほど良い)"]].mean().round(4).to_string())
