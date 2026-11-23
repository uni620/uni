import sys, logging
sys.path.insert(0, '/home/user/uni/keirin_ai'); logging.disable(logging.INFO)
import numpy as np, pandas as pd
from scipy.stats import binomtest
import keirin.model as M, keirin.features as F
from keirin.config import load_config; from keirin.db import DB; from keirin import pipeline
cfg = load_config(); db = DB('/home/user/uni/keirin_ai/data/keirin.sqlite')
f = pipeline.features_for(db, cfg); f = f[f.finish.notna()]
dates = sorted(f.date.unique())
base, ext = list(F.FEATURES), F.FEATURES + F.POS_RECORD_FEATURES + F.BANK_FEATURES
def run(train, test, feats, score):
    M.FEATURES = feats
    m = M.LGBModel(score).fit(train); p = m.predict(test)['p_win']
    t = test.assign(p=p.values)
    top = t.loc[t.groupby('race_id').p.idxmax()].set_index('race_id').finish.eq(1)
    ll = -np.log(t[t.finish == 1].set_index('race_id').p.clip(1e-6)).groupby(level=0).first()
    return top, ll
# 3つの検証期間（各約13日）: それより前で学習
for cut, end in [(dates[-39], dates[-26]), (dates[-26], dates[-13]), (dates[-13], None)]:
    train = f[f.date < cut]; test = f[(f.date >= cut) & ((f.date < end) if end else True)]
    for score in (False, True):
        a, la = run(train, test, base, score); b, lb = run(train, test, ext, score)
        n10, n01 = int((b & ~a).sum()), int((~b & a).sum())
        pv = binomtest(n10, n10 + n01).pvalue if n10 + n01 else 1
        print(f"{cut}〜 {'得点あり' if score else 'LightGBM'} races={len(a)} 1着率 今まで{a.mean():.3f} 位置+バンク{b.mean():.3f} "
              f"差{b.mean()-a.mean():+.3f} (新だけ当{n10}/旧だけ当{n01}, p={pv:.2f}) ズレ {la.mean():.3f}->{lb.mean():.3f}")
