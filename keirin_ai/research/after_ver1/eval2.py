import sys, logging
sys.path.insert(0, '/home/user/uni/keirin_ai')
logging.basicConfig(level=logging.WARNING)
import pandas as pd
from keirin.config import load_config
from keirin.db import DB
from keirin import pipeline
from keirin.model import evaluate
db_path, cut, end, train_from = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
cfg = load_config(); db = DB(db_path)
feats = pipeline.features_for(db, cfg)
feats = feats[feats["finish"].notna()]
train = feats[(feats.date < cut) & (feats.date >= train_from)]
test = feats[(feats.date >= cut) & (feats.date <= end)]
print(f"学習 {train.race_id.nunique()}R ({train.date.min()}〜) / 検証 {test.race_id.nunique()}R ({cut}〜{end})")
methods = pipeline.train_methods(train, pipeline.method_names(cfg))
rows = [{"予想方法": m.name, **evaluate(test, m.predict(test))} for m in methods]
pd.set_option('display.width', 250)
print(pd.DataFrame(rows).to_string(index=False))
db.con.execute("DELETE FROM bets WHERE mode='backtest'"); db.commit()
pipeline.predict_and_bet(db, cfg, methods, feats, test.race_id.unique().tolist(), "backtest")
pipeline.settle(db)
b = db.df("SELECT strategy, bet_type, SUM(stake) s, SUM(payout) p, SUM(status='won') w, COUNT(*) n FROM bets WHERE mode='backtest' GROUP BY 1,2")
b['回収率'] = (b.p / b.s * 100).round(1); b['的中率'] = (b.w / b.n * 100).round(1)
print(b[['strategy','bet_type','n','的中率','回収率']].to_string(index=False))
