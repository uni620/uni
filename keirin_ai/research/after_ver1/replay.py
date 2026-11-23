import sys, logging
sys.path.insert(0, '/home/user/uni/keirin_ai')
logging.basicConfig(level=logging.WARNING)
import pandas as pd
from keirin.config import load_config
from keirin.db import DB
from keirin import pipeline
D = "2026-11-19"
cfg = load_config(); db = DB(sys.argv[1])
rids = [r[0] for r in db.con.execute("SELECT DISTINCT race_id FROM bets WHERE mode='live' AND date=?", (D,))]
feats = pipeline.features_for(db, cfg)
train = feats[(feats.date < D) & feats.finish.notna()]
# 予想に使う行から結果の列を消す（念のため）
target = feats[feats.race_id.isin(rids)].copy()
target["finish"] = None; target["kimarite"] = None
print("学習", train.race_id.nunique(), "R（〜", train.date.max(), "） / 予想", target.race_id.nunique(), "R")
methods = pipeline.train_methods(train, pipeline.method_names(cfg))
db.con.execute("DELETE FROM bets WHERE mode='replay'"); db.commit()
pipeline.predict_and_bet(db, cfg, methods, target, rids, "replay")
pipeline.settle(db)
pd.set_option('display.width', 250)
b = db.df("""SELECT r.venue, r.race_no, b.strategy, b.bet_type, b.combo, b.status, b.payout FROM bets b JOIN races r USING(race_id)
             WHERE mode='replay' ORDER BY r.venue, r.race_no""")
b.to_csv(sys.argv[2], index=False)
a = db.df("SELECT mode, strategy, SUM(stake) s, SUM(payout) p, SUM(status='won') w FROM bets WHERE date=? AND mode IN ('live','replay') GROUP BY 1,2", (D,))
a['回収率'] = (a.p / a.s * 100).round(1); print(a.to_string(index=False))
t = db.df("SELECT mode, strategy, bet_type, SUM(stake) s, SUM(payout) p, SUM(status='won') w FROM bets WHERE date=? AND mode='replay' GROUP BY 1,2,3", (D,))
t['回収率'] = (t.p / t.s * 100).round(1); print(t.to_string(index=False))
