import sqlite3, numpy as np, pandas as pd
from scipy.optimize import minimize
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite',timeout=60)
T=pd.read_pickle(f'{S}/tenkai.pkl')[['race_id','car_no','p_tenkai']].merge(pd.read_pickle(f'{S}/tenkai2.pkl')[['race_id','car_no','date','finish','p_v2']],on=['race_id','car_no'])
P=pd.read_pickle(f'{S}/wf_art.pkl')
A=P[P.method=='+記事の要素'][['race_id','car_no','p']].rename(columns={'p':'p_art'})
B=P[P.method!='+記事の要素'][['race_id','car_no','p']].rename(columns={'p':'p_base'})
D=T.merge(A,on=['race_id','car_no']).merge(B,on=['race_id','car_no'])
D=D[D.date>='2026-09-01']
od=pd.read_sql("select race_id,combo,odds from odds where is_final=1 and bet_type='2車単'",c).drop_duplicates(['race_id','combo'])
od=od[od.race_id.isin(D.race_id)&(od.odds>0)]
fin=D.dropna(subset=['finish'])
w1=fin[fin.finish==1].groupby('race_id').car_no.agg(list); w2=fin[fin.finish==2].groupby('race_id').car_no.agg(list)
ok=[r for r in w1.index if len(w1[r])==1 and r in w2.index and len(w2[r])==1]
ans={r:f"{w1[r][0]}>{w2[r][0]}" for r in ok}
pay=pd.read_sql("select race_id,combo,payout from payouts where bet_type='2車単'",c).drop_duplicates(['race_id','combo'])
pay=dict(((r,k),v) for r,k,v in pay.itertuples(index=False))
AIS={'展開AI v1':'p_tenkai','展開AI v2':'p_v2','強さAI(LightGBM+競走得点+出走表)':'p_base','強さAI+記事の要素':'p_art'}
R=[]
for rid,g in od.groupby('race_id'):
    if rid not in ans: continue
    d=D[D.race_id==rid].set_index('car_no')
    a=g.combo.str.split('>').str[0].astype(int).values; b=g.combo.str.split('>').str[1].astype(int).values
    if not (set(a)|set(b))<=set(d.index): continue
    q=1/g.odds.values; q=q/q.sum()
    feats={'人気':np.log(q)}
    for lab,col in AIS.items():
        p=d[col]/d[col].sum(); pa=p.loc[a].values; pb=p.loc[b].values
        feats[lab]=np.log(np.clip(pa*pb/np.clip(1-pa,1e-6,None),1e-9,None))
    y=np.where(g.combo.values==ans[rid])[0]
    if len(y)!=1: continue
    R.append(dict(rid=rid,date=d.date.iloc[0],combo=g.combo.values,odds=g.odds.values,f=feats,y=y[0]))
print('レース数',len(R))
tr=[r for r in R if r['date']<'2026-10-01']; te=[r for r in R if r['date']>='2026-10-01']
def nll(w,RR,labs):
    o=0
    for r in RR:
        s=sum(wi*r['f'][l] for wi,l in zip(w,labs)); m=s.max(); o-=s[r['y']]-m-np.log(np.exp(s-m).sum())
    return o/len(RR)
def sim(w,RR,labs,ev_min,omax=10):
    n=hit=ret=0
    for r in RR:
        s=sum(wi*r['f'][l] for wi,l in zip(w,labs)); pm=np.exp(s-s.max()); pm/=pm.sum()
        ev=pm*r['odds']; ev[r['odds']>omax]=-1; i=ev.argmax()
        if ev[i]<ev_min: continue
        n+=1
        if i==r['y']: hit+=1; ret+=pay.get((r['rid'],r['combo'][i]),r['odds'][i]*100)
    return n,hit,ret/(n*100)*100 if n else 0
rows=[]
for lab in ['人気のみ']+list(AIS):
    labs=['人気'] if lab=='人気のみ' else ['人気',lab]
    w=minimize(lambda w:nll(w,tr,labs),np.r_[1.0,np.zeros(len(labs)-1)],method='L-BFGS-B').x
    for evm in (1.0,1.1,1.2):
        for nm,RR in (('9月(学習)',tr),('10/1〜(確認)',te)):
            n,h,roi=sim(w,RR,labs,evm); rows.append((lab,np.round(w,3).tolist(),evm,nm,n,h,round(roi,1)))
print(pd.DataFrame(rows,columns=['AI','比重','EV','期間','点数','的中','回収率%']).to_string())
