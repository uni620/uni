import sqlite3, numpy as np, pandas as pd
from scipy.optimize import minimize
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite',timeout=60)
T=pd.read_pickle(f'{S}/tenkai.pkl'); T=T[T.date>='2026-06-11']
P=pd.read_pickle(f'{S}/wf_art.pkl'); P=P[P.method=='+記事の要素'][['race_id','car_no','p']]
od=pd.read_sql("select race_id,combo,odds from odds where is_final=1 and bet_type='2車単'",c).drop_duplicates(['race_id','combo'])
od['car_no']=od.combo.str.split('>').str[0].astype(int); od['iq']=1/od.odds
Q=od.groupby(['race_id','car_no']).iq.sum().reset_index(); Q['q']=Q.iq/Q.groupby('race_id').iq.transform('sum')
D=T.merge(P,on=['race_id','car_no']).merge(Q[['race_id','car_no','q']],on=['race_id','car_no'])
D=D[D.groupby('race_id').car_no.transform('size')==D.groupby('race_id').car_no.transform('size')]
D['win']=(D.finish==1).astype(float)
ok=D.groupby('race_id').win.sum()==1; D=D[D.race_id.isin(ok[ok].index)]
for col,lab in (('p_tenkai','展開AI'),('p','今のAI（+記事の要素）'),('q','人気（2車単オッズ）')):
    top=D.loc[D.groupby('race_id')[col].idxmax()]
    print(f"{lab}: 本命の1着率 {top.win.mean()*100:.1f}%（{D.race_id.nunique()}R）")
print('展開AIと今のAIの本命が一致',round((D.loc[D.groupby('race_id').p_tenkai.idxmax()].car_no.values==D.loc[D.groupby('race_id').p.idxmax()].car_no.values).mean()*100,1),'%')
R=[]
for rid,g in D.groupby('race_id'):
    R.append((g.date.iloc[0],np.log(np.clip(g.q.to_numpy(),1e-6,1)),np.log(np.clip(g.p_tenkai.to_numpy(),1e-6,1)),np.log(np.clip(g.p.to_numpy()/g.p.sum(),1e-6,1)),int(np.argmax(g.win.to_numpy()))))
def per(w,RR,cols):
    o=[]
    for r in RR:
        s=sum(wi*r[ci] for wi,ci in zip(w,cols)); m=s.max(); o.append(-(s[r[4]]-m-np.log(np.exp(s-m).sum())))
    return np.array(o)
tr=[r for r in R if '2026-09-01'<=r[0]<'2026-10-01']; te=[r for r in R if r[0]>='2026-10-01']
rng=np.random.default_rng(0)
def fit(cols): return minimize(lambda w:per(w,tr,cols).mean(),np.r_[1.0,np.zeros(len(cols)-1)],method='L-BFGS-B').x
wm=fit([1]); base=per(wm,te,[1])
for lab,cols in (('人気＋展開AI',[1,2]),('人気＋今のAI',[1,3]),('人気＋今のAI＋展開AI',[1,3,2])):
    w=fit(cols); d=base-per(w,te,cols); bs=[d[rng.integers(0,len(d),len(d))].mean()*1000 for _ in range(1000)]
    print(f"{lab}: 比重 {np.round(w,3)} | 10/1〜の上乗せ {d.mean()*1000:+.2f}‰ [{np.percentile(bs,2.5):+.2f}〜{np.percentile(bs,97.5):+.2f}]")
