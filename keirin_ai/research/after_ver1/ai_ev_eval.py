import numpy as np, pandas as pd
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
T=pd.read_pickle(f'{S}/ai_ev_T.pkl')
rng=np.random.default_rng(0)
def ci(x):
    if len(x)<20: return "-"
    idx=rng.integers(0,len(x),(1000,len(x))); s=x[idx].sum(1)/(100*len(x))*100
    q=np.percentile(s,[2.5,97.5]); return f"{q[0]:.0f}-{q[1]:.0f}"
out=[]; picks={}
for cap in (10,20,30,1000):
    t=T[T.odds<=cap]
    best=t.loc[t.groupby(['method','race_id']).ev.idxmax()]
    for th in (0.8,1.0,1.2,1.5,2.0):
        b=best[best.ev>=th]
        for m,g in b.groupby('method'):
            h1=g[g.date<'2026-09-01']; h2=g[g.date>='2026-09-01']
            out.append(dict(方法=m,上限=cap,基準=th,点=len(g),的中率=round((g.pay>0).mean()*100,1),回収率=round(g.pay.sum()/len(g),1),
                CI=ci(g.pay.to_numpy()),前半=round(h1.pay.sum()/max(len(h1),1),0),後半=round(h2.pay.sum()/max(len(h2),1),0),
                予想当たり=round(g.p.sum(),0),実当たり=int((g.pay>0).sum())))
            picks[(m,cap,th)]=g
R=pd.DataFrame(out); pd.set_option('display.width',250)
print(R.sort_values('回収率',ascending=False).to_string(index=False))
R.to_csv(f'{S}/ai_ev_rules.csv',index=False)
# 選ばれた券種の内訳（代表）
for k in [("LightGBM+競走得点",30,1.2),("ライン連動",30,1.2)]:
    if k in picks: print(k, picks[k].bt.value_counts().to_dict())
