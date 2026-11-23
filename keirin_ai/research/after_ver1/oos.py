"""検証に使っていない古い期間（〜9/08）を、9/09〜11/19 で学習したモデルで予想する（時間を逆にした確認）。"""
import sys, logging, time
sys.path.insert(0, '/home/user/uni/keirin_ai')
logging.basicConfig(level=logging.WARNING)
import pandas as pd
from keirin.config import load_config
from keirin.db import DB
from keirin import pipeline
from keirin.model import make_methods
db_path, out = sys.argv[1], sys.argv[2]
names = ["LightGBM", "LightGBM+競走得点", "LightGBM+競走得点+位置バンク"]
cfg = load_config('/home/user/uni/keirin_ai/config.toml'); db = DB(db_path)
feats = pipeline.features_for(db, cfg); feats = feats[feats.finish.notna()]
train = feats[(feats.date >= "2026-09-09") & (feats.date <= "2026-11-19")]
test = feats[feats.date < "2026-09-09"]
parts = []
for m in make_methods(names):
    t0 = time.time(); m.fit(train); p = m.predict(test)["p_win"]
    parts.append(test[["race_id", "date", "car_no", "finish", "race_class", "grade", "venue", "start_time"]].assign(p=p.values, method=m.name, fold="oos"))
    print(m.name, "学習", train.race_id.nunique(), "R 検証", test.race_id.nunique(), f"R {time.time()-t0:.0f}s", flush=True)
pd.concat(parts).to_pickle(out)
