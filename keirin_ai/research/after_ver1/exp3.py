import sys, logging, pickle
sys.path.insert(0, '/home/user/uni/keirin_ai')
logging.basicConfig(level=logging.WARNING)
import pandas as pd, numpy as np, itertools
from keirin.config import load_config
from keirin.db import DB
from keirin import pipeline
from keirin.combos import combo_probs
cfg = load_config(); db = DB(sys.argv[1])
CUT, END = "2026-10-21", "2026-11-18"
feats = pipeline.features_for(db, cfg); feats = feats[feats.finish.notna()]
train = feats[feats.date < CUT]; test = feats[(feats.date >= CUT) & (feats.date <= END)]
ms = pipeline.train_methods(train, ["LightGBM+競走得点+位置バンク"])
p = ms[0].predict(test)["p_win"]
test = test.assign(p=p)
test[["race_id","date","car_no","p","race_class","grade"]].to_pickle(sys.argv[2])
