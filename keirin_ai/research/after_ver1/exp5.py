import sys, logging
sys.path.insert(0, '/home/user/uni/keirin_ai')
logging.basicConfig(level=logging.WARNING)
import pandas as pd, numpy as np
from keirin.config import load_config
from keirin.db import DB
from keirin import pipeline
from keirin.model import evaluate
cfg = load_config(); db = DB(sys.argv[1])
CUT, END = "2026-10-21", "2026-11-18"
feats = pipeline.features_for(db, cfg); feats = feats[feats.finish.notna()]
train = feats[feats.date < CUT]; test = feats[(feats.date >= CUT) & (feats.date <= END)]
names = ["LightGBM+競走得点+位置バンク", "LightGBM+競走得点+位置バンク+出走表"]
ms = pipeline.train_methods(train, names)
pd.set_option('display.width', 250)
for half, t in [("前", test[test.date < "2026-11-05"]), ("後", test[test.date >= "2026-11-05"])]:
    print(half, pd.DataFrame([{"方法": m.name, **evaluate(t, m.predict(t))} for m in ms]).iloc[:, :6].to_string(index=False))
db.con.execute("DELETE FROM bets WHERE mode='backtest'"); db.commit()
pipeline.predict_and_bet(db, cfg, ms, feats, test.race_id.unique().tolist(), "backtest")
pipeline.settle(db)
b = db.df("SELECT date, strategy, bet_type, status, payout, stake FROM bets WHERE mode='backtest'")
b["half"] = np.where(b.date < "2026-11-05", "前", "後")
o = b.groupby(["strategy", "bet_type", "half"]).agg(s=("stake", "sum"), p=("payout", "sum")).reset_index()
o["回収"] = (o.p / o.s * 100).round(1)
print(o.pivot_table(index=["strategy", "bet_type"], columns="half", values="回収").to_string())
imp = ms[1].importance(); print(imp.head(15).to_string(index=False))
