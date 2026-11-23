"""物差し: 1着の確率で「人気(2車単確定オッズ由来) と AI」を混ぜたとき、AIにどれだけ比重が乗るか。
P(i) ∝ exp(a·log q_i + b·log p_i)。b>0 が別期間でも残れば、AIに人気が知らない情報がある。"""
import sys, sqlite3, numpy as np, pandas as pd
from scipy.optimize import minimize
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
P=pd.read_pickle(sys.argv[1] if len(sys.argv)>1 else f'{S}/wf_all.pkl')
c=sqlite3.connect(f'{S}/sim.sqlite')
od=pd.read_sql("select race_id,combo,odds,fetched_at from odds where is_final=1 and bet_type='2車単' and race_id>='20260611'",c)
od=od.sort_values('fetched_at').drop_duplicates(['race_id','combo'],keep='last')
od['f']=od.combo.str.split('>').str[0].astype(int); od['iq']=1/od.odds
Q=od.groupby(['race_id','f']).iq.sum().rename('q').reset_index().rename(columns={'f':'car_no'})
Q['q']=Q.q/Q.groupby('race_id').q.transform('sum')
rc=pd.read_sql("select race_id,race_class,grade,start_time from races",c)
def races_for(m):
    d=P[P.method==m].merge(Q,on=['race_id','car_no'])
    d=d[d.groupby('race_id').car_no.transform('size')==P[P.method==m].groupby('race_id').car_no.transform('size').reindex(d.index).fillna(0)] if False else d
    out=[]
    for rid,g in d.groupby('race_id'):
        if (g.finish==1).sum()!=1 or len(g)<5: continue
        p=g.p.to_numpy(); p=np.clip(p/p.sum(),1e-6,1); q=np.clip(g.q.to_numpy(),1e-6,1)
        out.append((rid,g.date.iloc[0],np.log(q),np.log(p),int(np.argmax(g.finish.to_numpy()==1))))
    return out
def nll(w,R):
    t=0
    for _,_,lq,lp,y in R:
        s=w[0]*lq+w[1]*lp; m=s.max(); t-=s[y]-m-np.log(np.exp(s-m).sum())
    return t/len(R)
def fit(R): return minimize(nll,[1.0,0.0],args=(R,),method='L-BFGS-B').x
rng=np.random.default_rng(0)
for m in (sys.argv[2].split(",") if len(sys.argv)>2 else P.method.unique()):
    R=races_for(m)
    tr=[r for r in R if r[1]<'2026-09-01']; te=[r for r in R if r[1]>='2026-09-01']
    w=fit(R)
    bs=[fit([R[i] for i in rng.integers(0,len(R),len(R))])[1] for _ in range(30)]
    wt=fit(tr)
    ll_m=nll([1,0],te); ll_a=nll([0,1],te); ll_b=nll(wt,te); ll_mf=nll(fit([(a,b,c_,d*0,y) for a,b,c_,d,y in tr]),te)
    print(f"{m}: レース{len(R)} 比重 a(人気)={w[0]:.3f} b(AI)={w[1]:.3f} [{np.percentile(bs,2.5):.3f}〜{np.percentile(bs,97.5):.3f}] | 後半の対数損失 人気{ll_m:.4f} AI{ll_a:.4f} 混合(前半で学習){ll_b:.4f} 差{(ll_m-ll_b)*1000:+.1f}‰ 人気だけ調整{ll_mf:.4f} AIの上乗せ{(ll_mf-ll_b)*1000:+.1f}‰",flush=True)
