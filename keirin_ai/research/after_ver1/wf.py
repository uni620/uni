"""期間をずらして何回も学習・予想し、各選手の1着確率を保存する（ウォークフォワード検証）。"""
import sys, logging, time
sys.path.insert(0, '/home/user/uni/keirin_ai')
logging.basicConfig(level=logging.WARNING)
import pandas as pd
from keirin.config import load_config
from keirin.db import DB
from keirin import pipeline
from keirin.model import make_methods
db_path, out, names = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
FOLDS = [("2026-09-09", "2026-09-22"), ("2026-09-23", "2026-10-06"), ("2026-10-07", "2026-10-20"), ("2026-10-21", "2026-11-03"), ("2026-11-04", "2026-11-19")]
cfg = load_config(); db = DB(db_path)
feats = pipeline.features_for(db, cfg); feats = feats[feats.finish.notna()]
parts = []
for a, b in FOLDS:
    train = feats[feats.date < a]; test = feats[(feats.date >= a) & (feats.date <= b)]
    for m in make_methods(names):
        t0 = time.time(); m.fit(train)
        p = m.predict(test)["p_win"]
        parts.append(test[["race_id", "date", "car_no", "finish", "race_class", "grade", "venue", "start_time"]].assign(p=p.values, method=m.name, fold=a))
        print(a, m.name, "学習", train.race_id.nunique(), "R 検証", test.race_id.nunique(), f"R {time.time()-t0:.0f}s", flush=True)
pd.concat(parts).to_pickle(out)
