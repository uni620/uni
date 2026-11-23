"""手順5の準備: 全レースの組み合わせ表（オッズ特徴 + ライン連動AIの確率pl + 実際の払戻）。"""
import pickle, sqlite3, numpy as np, pandas as pd, sys
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'; sys.path.insert(0,S); sys.path.insert(0,'/home/user/uni/keirin_ai')
from step3_fit import combos, R
from keirin.market_model import market_table
cols,w2,w3=pickle.load(open(f'{S}/step3_w.pkl','rb'))['基本7']
c=sqlite3.connect(f'{S}/sim.sqlite')
od=pd.read_sql("select race_id,bet_type,combo,odds,fetched_at from odds where is_final=1 and race_id>='20260611'",c)
od=od.sort_values('fetched_at').drop_duplicates(['race_id','bet_type','combo'],keep='last')
OD={}
for (rid,bt),g in od.groupby(['race_id','bet_type']): OD.setdefault(rid,{})[bt]=dict(zip(g.combo,g.odds))
pay=pd.read_sql("select race_id,bet_type,combo,payout from payouts",c)
PAY={(r.race_id,r.bet_type,r.combo):r.payout for r in pay.itertuples()}
meta=pd.read_sql("select race_id,start_time from races",c).set_index('race_id').start_time.to_dict()
out=[]
for r in R:
    pr,_=combos(r,w2,w3,cols)
    t=market_table(OD[r['rid']],r['n'],pr)
    if t is None: continue
    t['race_id']=r['rid']; t['date']=r['date']; t['hour']=int(str(meta.get(r['rid']) or '00:00')[:2])
    t['pay']=[PAY.get((r['rid'],b,cb),0) for b,cb in zip(t.bt,t.combo)]
    out.append(t)
T=pd.concat(out,ignore_index=True); T.to_pickle(f'{S}/step5_T.pkl'); print(len(T),T.race_id.nunique())
