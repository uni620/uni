"""AI確率×確定オッズで、レースごとに4券種から期待値が一番高い1点を選ぶ（基準未満は見送り）。"""
import sys, sqlite3, numpy as np, pandas as pd
sys.path.insert(0,'/home/user/uni/keirin_ai')
from keirin.combos import combo_probs
from keirin.line_combos import line_combo_probs
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect(f'{S}/sim.sqlite')
P=pd.read_pickle(f'{S}/wf_all.pkl')
od=pd.read_sql("select race_id,bet_type,combo,odds,fetched_at from odds where is_final=1 and race_id>='20260611'",c)
od=od.sort_values('fetched_at').drop_duplicates(['race_id','bet_type','combo'],keep='last')
OD={k:dict(zip(g.combo,g.odds)) for k,g in od.groupby(['race_id','bet_type'])}
pay=pd.read_sql("select race_id,bet_type,combo,payout from payouts",c)
PAY={(r.race_id,r.bet_type,r.combo):r.payout for r in pay.itertuples()}
scr=set(pd.read_sql("select distinct race_id from entries where scratched=1",c).race_id)
BTS=["2車複","ワイド","2車単","3連複"]
rows=[]
meths=list(P.method.unique())+["ライン連動"]
for m in meths:
    src=P[P.method==("LightGBM+競走得点" if m=="ライン連動" else m)]
    for rid,g in src.groupby('race_id'):
        if rid in scr or any((rid,bt) not in OD for bt in BTS): continue
        g=g.sort_values('car_no'); cars=g.car_no.astype(int).tolist(); p=g.p.to_numpy(); p=p/p.sum()
        if m=="ライン連動":
            lines={int(r.car_no):(r.line_id,r.line_pos,r.line_size) for r in g.itertuples() if pd.notna(r.line_id)}
            pr=line_combo_probs(cars,p,lines)
        else:
            pr=combo_probs(cars,p)
        for bt in BTS:
            o=OD[(rid,bt)]
            for cb,q in pr[bt].items():
                if cb in o and o[cb]>0:
                    rows.append((m,rid,g.date.iloc[0],bt,cb,q,o[cb]))
T=pd.DataFrame(rows,columns=['method','race_id','date','bt','combo','p','odds'])
T['ev']=T.p*T.odds
T['pay']=[PAY.get((r,b,cb),0) for r,b,cb in zip(T.race_id,T.bt,T.combo)]
T.to_pickle(f'{S}/ai_ev_T.pkl')
print(T.groupby('method').race_id.nunique())
