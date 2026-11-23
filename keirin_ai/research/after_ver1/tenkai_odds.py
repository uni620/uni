"""展開AIの確率をver1のオッズモデルに足す。展開AIは9/1以降が学習外なので、オッズモデルは 9/1〜10/10 で学習し 10/11〜11/20 で確かめる。
比較用に、同じ期間で学習した「オッズだけ」も作る。"""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/user/uni/keirin_ai')
from scipy.optimize import minimize
from keirin.combos import combo_probs
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
T=pd.read_pickle(f'{S}/step5_T.pkl'); T=T[T.date>='2026-09-01']
K=pd.read_pickle(f'{S}/tenkai.pkl'); K=K[K.date>='2026-09-01']
PT={}
for rid,g in K.groupby('race_id'):
    g=g.sort_values('car_no'); pr=combo_probs(g.car_no.astype(int).tolist(),g.p_tenkai.to_numpy())
    for bt,d in pr.items():
        for cb,v in d.items(): PT[(rid,bt,cb)]=v
T['pt']=[PT.get((r,b,cb),np.nan) for r,b,cb in zip(T.race_id,T.bt,T.combo)]
T=T.dropna(subset=['pt'])
COLS={'2車複':['q','q_ex','q_tr','q_qu','q_wi'],'ワイド':['q','q_ex','q_tr','q_qu','q_wi'],'2車単':['q','q_ex','q_tr','q_qu','q_wi','share'],'3連複':['q','s_wi','s_qu','s_ex','w3']}
CUT='2026-10-11'
class LR:
    def fit(self,X,y,C=1.0):
        self.mu=X.mean(0); self.sd=X.std(0)+1e-9; Z=np.column_stack([np.ones(len(X)),(X-self.mu)/self.sd])
        def f(w):
            z=Z@w; p=1/(1+np.exp(-z)); l=-(y*np.log(p+1e-12)+(1-y)*np.log(1-p+1e-12)).sum()+0.5/C*(w[1:]**2).sum()
            g=Z.T@(p-y); g[1:]+=w[1:]/C; return l,g
        self.w=minimize(f,np.zeros(Z.shape[1]),jac=True,method='L-BFGS-B').x; return self
    def p(self,X): return 1/(1+np.exp(-(np.column_stack([np.ones(len(X)),(X-self.mu)/self.sd])@self.w)))
res={}
for mname,extra in (('オッズだけ',False),('オッズ+展開AI',True)):
    parts=[]
    for bt,cols in COLS.items():
        x=T[T.bt==bt].dropna(subset=cols).copy()
        F=[np.log(x[c].astype(float).clip(lower=1e-9)) for c in cols]+[np.log(x.odds.astype(float)),x.nine.astype(float)]
        if extra: F.append(np.log(x.pt.clip(lower=1e-9)))
        F=np.column_stack(F); y=(x.pay>0).astype(float).to_numpy(); tr=(x.date<CUT).to_numpy()
        m=LR().fit(F[tr],y[tr]); x['pm']=m.p(F); x['ev']=x.pm*x.odds
        if extra: print(bt,'展開AIの係数(標準化)',round(m.w[-1],3))
        parts.append(x[['race_id','date','bt','combo','odds','pay','pm','ev']])
    res[mname]=pd.concat(parts)
rng=np.random.default_rng(0)
def summ(b):
    if len(b)<15: return f"{len(b)}点"
    v=b.pay.to_numpy(float); i=rng.integers(0,len(v),(2000,len(v))); s=v[i].mean(1)
    return f"{len(b)}点 的中{(v>0).sum()} {v.mean():.0f}% [{np.percentile(s,2.5):.0f}〜{np.percentile(s,97.5):.0f}]"
for mname,x in res.items():
    te=x[x.date>=CUT]
    ll=-(np.log(np.where(te.pay>0,te.pm,1-te.pm).clip(1e-12))).mean()
    print(f"== {mname}（確認 10/11〜） 外れ具合 {ll:.5f}")
    for wide in ('あり','なし'):
        t=te if wide=='あり' else te[te.bt!='ワイド']
        out=[]
        for cap,th in ((10,1.1),(10,1.2),(30,1.2)):
            u=t[t.odds<=cap]; best=u.loc[u.groupby('race_id').ev.idxmax()]; out.append(f"{cap}倍・{th}: {summ(best[best.ev>=th])}")
        print(f"  ワイド{wide}: "+" | ".join(out))
