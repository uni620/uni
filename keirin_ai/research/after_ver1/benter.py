"""見直し案: 実力のモデル（頭AI, オッズなし）× 人気 を合体（条件付きロジット）。事前登録ルールで9〜11月を月ごとに確認。
事前登録: 2車単 1R1点 EV>=1.1 かつ 30倍以下 / 3連単 1R1点 EV>=1.1 かつ 100倍以下。合格 = 3か月とも100%超 かつ 合計の95%下限>100%。"""
import sqlite3,itertools,pandas as pd,numpy as np
from scipy.optimize import minimize
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
H=pd.read_pickle(f'{S}/hx_conf.pkl')          # 9〜11月、各月その月より前で学習した頭AIの確率 pr
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
rids=set(H.race_id)
od=pd.read_sql("select race_id,bet_type,combo,odds from odds where is_final=1 and bet_type in ('2車単','3連単') and odds>0",c)
od=od[od.race_id.isin(rids)].drop_duplicates(['race_id','bet_type','combo'])
pay=pd.read_sql("select race_id,bet_type,combo,payout from payouts where bet_type in ('2車単','3連単')",c)
PAY={(r,b,k):v for r,b,k,v in pay.itertuples(index=False)}
fin=pd.read_sql("select race_id,car_no,finish from entries where finish<=3",c); fin=fin[fin.race_id.isin(rids)]
ANS={}
for r,g in fin.groupby('race_id'):
    g=g.sort_values('finish')
    if list(g.finish)==[1,2,3]: c3=g.car_no.astype(int).tolist(); ANS[(r,'2車単')]=f'{c3[0]}>{c3[1]}'; ANS[(r,'3連単')]=f'{c3[0]}>{c3[1]}>{c3[2]}'
OD={(r,b):dict(zip(g.combo,g.odds)) for (r,b),g in od.groupby(['race_id','bet_type'])}
recs={'2車単':[],'3連単':[]}
for rid,g in H.groupby('race_id'):
    p=dict(zip(g.car_no.astype(int),g.pr/g.pr.sum())); cars=list(p); m=g.date.iloc[0][:7]
    for bt in recs:
        O=OD.get((rid,bt)); a=ANS.get((rid,bt))
        if not O or a is None or a not in O: continue
        ks,ai=[],[]
        it=itertools.permutations(cars,2 if bt=='2車単' else 3)
        for t in it:
            k='>'.join(map(str,t))
            if k not in O: continue
            v=p[t[0]]*p[t[1]]/max(1-p[t[0]],1e-6)
            if bt=='3連単': v*=p[t[2]]/max(1-p[t[0]]-p[t[1]],1e-6)
            ks.append(k); ai.append(v)
        o=np.array([O[k] for k in ks]); q=(1/o)/(1/o).sum(); ai=np.array(ai); ai=ai/ai.sum()
        recs[bt].append(dict(rid=rid,m=m,ks=ks,o=o,lq=np.log(q),la=np.log(np.clip(ai,1e-9,1)),y=ks.index(a)))
def nll(w,R,use_ai):
    t=0
    for r in R:
        s=w[0]*r['lq']+(w[1]*r['la'] if use_ai else 0); s=s-s.max(); t-= s[r['y']]-np.log(np.exp(s).sum())
    return t/len(R)
rng=np.random.default_rng(0)
months=['2026-09','2026-10','2026-11']
for bt,(cap,thr) in (('2車単',(30,1.1)),('3連単',(100,1.1))):
    R=recs[bt]; print(f'===== {bt}  レース数 {len(R)}')
    out=[]; gain=[]
    for i,mm in enumerate(months):
        tr=[r for r in R if r['m']<mm] if i>0 else [r for r in R if r['m']==mm]   # 9月は9月の中で（学習期間）
        te=[r for r in R if r['m']==mm]
        w0=minimize(lambda w:nll(w,tr,False),[1.0,0],method='L-BFGS-B').x
        w=minimize(lambda w:nll(w,tr,True),[1.0,0.1],method='L-BFGS-B').x
        gain.append((nll(w0,te,False)-nll(w,te,True))*1000)
        for r in te:
            s=w[0]*r['lq']+w[1]*r['la']; P=np.exp(s-s.max()); P/=P.sum(); ev=P*r['o']; ev[r['o']>cap]=-1; j=ev.argmax()
            if ev[j]>=thr: out.append((mm,r['rid'],r['ks'][j],r['o'][j],ev[j],PAY.get((r['rid'],bt,r['ks'][j]),0)))
        print(f'  {mm}: 比重 人気a={w[0]:.3f} AIb={w[1]:.3f}  人気だけより {gain[-1]:+.2f}‰（{"学習期間＝同じ月で決定" if i==0 else "学習外"}）')
    X=pd.DataFrame(out,columns=['月','rid','combo','odds','ev','pay'])
    for mm in months:
        x=X[X.月==mm]; print(f'  {mm}: {len(x)}点 的中{(x.pay>0).sum()} 回収率 {x.pay.mean() if len(x) else 0:.1f}%')
    x=X[X.月>'2026-09']; v=x.pay.values; s=[v[rng.integers(0,len(v),len(v))].mean() for _ in range(2000)] if len(v) else [0]
    print(f'  学習外（10〜11月）合計: {len(v)}点 回収率 {v.mean() if len(v) else 0:.1f}% 95%幅 {np.percentile(s,2.5):.0f}〜{np.percentile(s,97.5):.0f}  平均オッズ {x.odds.mean():.1f}')
