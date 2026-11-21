"""9〜11月（walk-forward予想）× Kドリームス確定オッズで、オッズを使った買い方を検証する。
出力: odds_val.pkl（組み合わせ単位の表）と、画面に各集計。"""
import sys, sqlite3, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/uni/keirin_ai')
from keirin.combos import combo_probs
S = '/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c = sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
BTS = ['2車複', '2車単', '3連複', 'ワイド']

t = pd.read_pickle(f'{S}/wf_v2.pkl')
od = pd.read_sql("select race_id,bet_type,combo,odds from odds where is_final=1", c)
od = od[od.race_id.isin(set(t.race_id))]
scr = set(pd.read_sql("select race_id from entries where scratched=1", c).race_id)
pay = pd.read_sql("select race_id,bet_type,combo,payout from payouts", c)
P = {(r.race_id, r.bet_type, r.combo): r.payout for r in pay.itertuples()}
O = {}
for r in od.itertuples():
    O.setdefault((r.race_id, r.bet_type), {})[r.combo] = r.odds
ok = {rid for (rid, bt) in O if all((rid, b) in O for b in BTS)} - scr
ok &= {rid for (rid, bt, cb) in P if bt == 'ワイド'}
rows = []
for (m, rid), g in t[t.race_id.isin(ok)].groupby(['method', 'race_id']):
    cars = g.car_no.astype(int).tolist(); p = g.p.to_numpy()
    cp = combo_probs(cars, p)
    for bt in BTS:
        o = O[(rid, bt)]
        for cb, pr in cp[bt].items():
            if cb not in o:
                continue
            rows.append((m, rid, g.date.iloc[0], g.race_class.iloc[0], g.grade.iloc[0], len(cars), bt, cb, pr, o[cb], P.get((rid, bt, cb), 0)))
d = pd.DataFrame(rows, columns=['method', 'race_id', 'date', 'cls', 'grade', 'nr', 'bt', 'combo', 'p', 'odds', 'ret'])
# 市場の確率: 1/オッズ をレース・券種ごとに合計1（ワイドは3）にそろえる
d['q0'] = 1 / d.odds
d['q'] = d.q0 / d.groupby(['method', 'race_id', 'bt']).q0.transform('sum') * np.where(d.bt == 'ワイド', 3, 1)
d['ev'] = d.p * d.odds
d['rk_p'] = d.groupby(['method', 'race_id', 'bt']).p.rank(ascending=False, method='first')
d['rk_q'] = d.groupby(['method', 'race_id', 'bt']).odds.rank(method='first')
d.to_pickle(f'{S}/odds_val.pkl')
print('レース数', d.race_id.nunique(), d.date.min(), d.date.max(), '行', len(d))
