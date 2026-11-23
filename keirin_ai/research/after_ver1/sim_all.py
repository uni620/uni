"""取れた全期間で、本番と同じ予想・買い方を過去レースに当てはめる（月ごとに学習し直すウォークフォワード）。"""
import sys, time, logging, sqlite3
sys.path.insert(0, '/home/user/uni/keirin_ai')
logging.basicConfig(level=logging.WARNING)
import pandas as pd
from keirin.config import load_config
from keirin.db import DB
from keirin import pipeline
S = '/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
src = sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite'); dst = sqlite3.connect(f'{S}/sim.sqlite')
src.backup(dst); dst.close(); src.close()
cfg = load_config('/home/user/uni/keirin_ai/config.toml'); db = DB(f'{S}/sim.sqlite')
db.con.execute("DELETE FROM bets WHERE mode='sim'"); db.commit()
FOLDS = [("2026-05-11", "2026-06-10"), ("2026-06-11", "2026-07-10"), ("2026-07-11", "2026-08-10"),
         ("2026-08-11", "2026-09-10"), ("2026-09-11", "2026-10-10"), ("2026-10-11", "2026-11-20")]
t0 = time.time()
feats = pipeline.features_for(db, cfg); feats = feats[feats.finish.notna()]
print("特徴量", feats.race_id.nunique(), "R", feats.date.min(), "〜", feats.date.max(), f"{time.time()-t0:.0f}s", flush=True)
names = pipeline.method_names(cfg)
for a, b in FOLDS:
    train = feats[feats.date < a]
    test = feats[(feats.date >= a) & (feats.date <= b)].copy()
    test["finish"] = None
    if "kimarite" in test: test["kimarite"] = None
    methods = pipeline.train_methods(train, names)
    n = pipeline.predict_and_bet(db, cfg, methods, test, test.race_id.unique().tolist(), "sim")
    print(a, "学習", train.race_id.nunique(), "R / 検証", test.race_id.nunique(), "R /", n, "点", f"{time.time()-t0:.0f}s", flush=True)
print("精算", pipeline.settle(db), flush=True)
