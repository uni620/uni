import sys, logging
sys.path.insert(0, '/home/user/uni/keirin_ai')
logging.basicConfig(level=logging.WARNING)
import pandas as pd
from keirin.config import load_config
from keirin.db import DB
from keirin import pipeline
from keirin.report import summary_table
import keirin.model as M, keirin.features as F
extra = sys.argv[2].split(',') if len(sys.argv) > 2 and sys.argv[2] else []
pass
cfg = load_config()
db = DB(sys.argv[1])
t = pipeline.evaluate_split(db, cfg, test_frac=0.22)
pd.set_option('display.width', 250)
print(t.to_string(index=False))
b = db.df("SELECT strategy, bet_type, SUM(stake) s, SUM(payout) p, SUM(status='won') w, COUNT(*) n FROM bets WHERE mode='backtest' GROUP BY 1,2")
b['回収率'] = (b.p / b.s * 100).round(1); b['的中率'] = (b.w / b.n * 100).round(1)
print(b[['strategy','bet_type','n','的中率','回収率']].to_string(index=False))
