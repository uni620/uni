import sqlite3, numpy as np, pandas as pd
from scipy.optimize import minimize
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite',timeout=60)
T=pd.read_pickle(f'{S}/tenkai.pkl')[['race_id','car_no','p_tenkai']].merge(pd.read_pickle(f'{S}/tenkai2.pkl')[['race_id','car_no','date','finish','p_v2']],on=['race_id','car_no'])
T=T[T.date>='2026-06-11']
P=pd.read_pickle(f'{S}/wf_art.pkl'); P=P[P.method=='+記事の要素'][['race_id','car_no','p']]
od=pd.read_sql("select race_id,combo,odds from odds where is_final=1 and bet_type='2車単'",c).drop_duplicates(['race_id','combo'])
od['car_no']=od.combo.str.split('>').str[0].astype(int); od['iq']=1/od.odds
Q=od.groupby(['race_id','car_no']).iq.sum().reset_index(); Q['q']=Q.iq/Q.groupby('race_id').iq.transform('sum')
D=T.merge(P,on=['race_id','car_no']).merge(Q[['race_id','car_no','q']],on=['race_id','car_no'])
D['win']=(D.finish==1).astype(float); ok=D.groupby('race_id').win.sum()==1; D=D[D.race_id.isin(ok[ok].index)]
for col,lab in (('p_v2','展開AI v2'),('p_tenkai','展開AI v1'),('p','今のAI'),('q','人気')):
    top=D.loc[D.groupby('race_id')[col].idxmax()]; print(f"{lab}: 本命の1着率 {top.win.mean()*100:.1f}%（{D.race_id.nunique()}R、6/11〜。展開AIは9/1〜が学習外）")
R=[]
for rid,g in D.groupby('race_id'):
    n=lambda v:np.log(np.clip(v/v.sum(),1e-6,1))
    R.append((g.date.iloc[0],n(g.q.to_numpy()),n(g.p_tenkai.to_numpy()),n(g.p_v2.to_numpy()),n(g.p.to_numpy()),int(np.argmax(g.win.to_numpy()))))
def per(w,RR,cols):
    o=[]
    for r in RR:
        s=sum(wi*r[ci] for wi,ci in zip(w,cols)); m=s.max(); o.append(-(s[r[5]]-m-np.log(np.exp(s-m).sum())))
    return np.array(o)
tr=[r for r in R if '2026-09-01'<=r[0]<'2026-10-01']; te=[r for r in R if r[0]>='2026-10-01']
rng=np.random.default_rng(0)
fit=lambda cols: minimize(lambda w:per(w,tr,cols).mean(),np.r_[1.0,np.zeros(len(cols)-1)],method='L-BFGS-B').x
base=per(fit([1]),te,[1])
for lab,cols in (('人気＋展開AI v2',[1,3]),('人気＋展開AI v1',[1,2]),('人気＋今のAI',[1,4])):
    w=fit(cols); d=base-per(w,te,cols); bs=[d[rng.integers(0,len(d),len(d))].mean()*1000 for _ in range(1000)]
    print(f"{lab}: 比重 {np.round(w,3)} | 10/1〜の上乗せ {d.mean()*1000:+.2f}‰ [{np.percentile(bs,2.5):+.2f}〜{np.percentile(bs,97.5):+.2f}]")
