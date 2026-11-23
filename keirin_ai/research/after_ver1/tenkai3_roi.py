"""展開AI v3 の回収率（確定オッズ・10/1〜11/20、学習外）。
各組み合わせの『ずれ比』= v3 の確率 ÷ 人気の確率（どちらも同じ並べ方 Harville）。期待値はずれ比に比例するので、
1レース1点（ずれ比が最大の目）をずれ比が基準以上なら買う。"""
import sqlite3, itertools, numpy as np, pandas as pd
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite',timeout=60)
E=pd.read_pickle(f'{S}/tenkai3.pkl'); E=E[E.date>='2026-10-01']
rids=set(E.race_id)
od=pd.read_sql("select race_id,bet_type bt,combo,odds from odds where is_final=1 and odds>0 and bet_type in ('2車単','2車複','3連複','3連単')",c)
od=od[od.race_id.isin(rids)].drop_duplicates(['race_id','bt','combo'])
pay=pd.read_sql("select race_id,bet_type bt,combo,payout from payouts",c); pay=pay[pay.race_id.isin(rids)]
PAY={(r,b,k):v for r,b,k,v in pay.itertuples(index=False)}
def probs(cars,p):
    p=np.asarray(p,float); n=len(p); o={'2車単':{},'2車複':{},'3連複':{},'3連単':{}}
    for i,j,k in itertools.permutations(range(n),3):
        v=p[i]*p[j]/(1-p[i])*p[k]/(1-p[i]-p[j]); a,b,d=cars[i],cars[j],cars[k]
        o['3連単'][f'{a}>{b}>{d}']=v; s='-'.join(map(str,sorted((a,b,d)))); o['3連複'][s]=o['3連複'].get(s,0)+v
    for i,j in itertools.permutations(range(n),2):
        v=p[i]*p[j]/(1-p[i]); o['2車単'][f'{cars[i]}>{cars[j]}']=v; s=f'{min(cars[i],cars[j])}-{max(cars[i],cars[j])}'; o['2車複'][s]=o['2車複'].get(s,0)+v
    return o
rows=[]
for rid,g in E.groupby('race_id'):
    g=g.sort_values('car_no'); cars=g.car_no.astype(int).tolist()
    A=probs(cars,g.p3.values); B=probs(cars,g.base.values)
    for bt in A:
        for k,v in A[bt].items(): rows.append((rid,g.date.iloc[0],bt,k,v/B[bt][k]))
R=pd.DataFrame(rows,columns=['race_id','date','bt','combo','ratio']).merge(od,on=['race_id','bt','combo'])
R['pay']=[PAY.get((r,b,k),0) for r,b,k in zip(R.race_id,R.bt,R.combo)]
R.to_pickle(f'{S}/tenkai3_R.pkl')
rng=np.random.default_rng(0)
def show(x,lab):
    out=[]
    for cap in (10,30,100):
        t=x[x.odds<=cap]
        if t.empty: continue
        best=t.loc[t.groupby('race_id').ratio.idxmax()]
        for th in (1.1,1.2,1.3,1.5):
            b=best[best.ratio>=th]; v=b.pay.to_numpy(float)
            if len(v)<20: continue
            s=v[rng.integers(0,len(v),(2000,len(v)))].mean(1)
            out.append((lab,cap,th,len(v),int((v>0).sum()),round(v.mean(),1),f'{np.percentile(s,2.5):.0f}〜{np.percentile(s,97.5):.0f}'))
    return out
res=[]
for bt in ('2車単','2車複','3連複','3連単'): res+=show(R[R.bt==bt],bt)
res+=show(R,'4券種から1点')
pd.set_option('display.width',200); pd.set_option('display.max_rows',300)
print(pd.DataFrame(res,columns=['券種','上限倍','ずれ比','点','的中','回収率%','95%幅']).to_string(index=False))
