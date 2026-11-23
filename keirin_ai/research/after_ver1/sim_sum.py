import sqlite3, numpy as np, pandas as pd
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect(f'{S}/sim.sqlite')
b=pd.read_sql("select race_id,date,strategy,bet_type,stake,payout,status from bets where mode='sim' and status in ('won','lost','refund')",c)
b['hit']=b.status=='won'; b['m']=b.date.str[:7]
rng=np.random.default_rng(0)
def ci(g):
    r=g.groupby('race_id')[['stake','payout']].sum().to_numpy()
    idx=rng.integers(0,len(r),(1000,len(r))); s=r[idx].sum(1)
    q=np.percentile(s[:,1]/s[:,0]*100,[2.5,97.5]); return f"{q[0]:.0f}-{q[1]:.0f}"
rows=[]
for (s,bt),g in b.groupby(['strategy','bet_type']):
    rows.append(dict(方法=s,券種=bt,点=len(g),的中率=round(g.hit.mean()*100,1),回収率=round(g.payout.sum()/g.stake.sum()*100,1),CI=ci(g)))
T=pd.DataFrame(rows).sort_values('回収率',ascending=False)
pd.set_option('display.width',250); print(T.to_string(index=False))
M=b.groupby(['strategy','m']).apply(lambda g: round(g.payout.sum()/g.stake.sum()*100)).unstack()
print(M.to_string())
T.to_csv(f'{S}/sim_table.csv',index=False); M.to_csv(f'{S}/sim_month.csv')
