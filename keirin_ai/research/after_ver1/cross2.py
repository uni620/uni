"""オッズだけで作る確率（券種どうしのずれを使う）→ 期待値 → 回収率。前半で学習、後半で確認。"""
import pandas as pd, numpy as np, sys
from scipy.optimize import minimize
class LogisticRegression:
    def __init__(self,C=1.0,max_iter=2000): self.C=C
    def fit(self,X,y):
        self.mu=X.mean(0); self.sd=X.std(0)+1e-9; Z=(X-self.mu)/self.sd; Z=np.column_stack([np.ones(len(Z)),Z])
        def f(w):
            z=Z@w; p=1/(1+np.exp(-z)); l=-(y*np.log(p+1e-12)+(1-y)*np.log(1-p+1e-12)).sum()+0.5/self.C*(w[1:]**2).sum()
            g=Z.T@(p-y); g[1:]+=w[1:]/self.C; return l,g
        self.w=minimize(f,np.zeros(Z.shape[1]),jac=True,method='L-BFGS-B').x
        self.coef_=[self.w[1:]/self.sd]; return self
    def predict_proba(self,X):
        Z=np.column_stack([np.ones(len(X)),(X-self.mu)/self.sd]); p=1/(1+np.exp(-(Z@self.w))); return np.column_stack([1-p,p])
CUT=sys.argv[1] if len(sys.argv)>1 else '2026-10-15'
d=pd.read_pickle('odds_val.pkl'); d=d[d.method=='LightGBM'][['race_id','date','cls','nr','bt','combo','odds','ret','q0']].copy()
d['q']=d.q0/d.groupby(['race_id','bt']).q0.transform('sum')
pair=lambda s:'-'.join(sorted(s.replace('>','-').split('-'),key=int))
ex=d[d.bt=='2車単'].copy(); ex['pair']=ex.combo.map(pair)
qex=ex.groupby(['race_id','pair']).q.sum().rename('q_ex')
tr=d[d.bt=='3連複']; rows=[]
for rid,cb,q in zip(tr.race_id,tr.combo,tr.q):
    a,b,c=cb.split('-'); rows+= [(rid,f"{a}-{b}",q),(rid,f"{a}-{c}",q),(rid,f"{b}-{c}",q)]
qtr=pd.DataFrame(rows,columns=['race_id','pair','q_tr']).groupby(['race_id','pair']).q_tr.sum()
qqu=d[d.bt=='2車複'].set_index(['race_id','combo']).q.rename('q_qu'); qqu.index.names=['race_id','pair']
qwi=d[d.bt=='ワイド'].set_index(['race_id','combo']).q.rename('q_wi'); qwi.index.names=['race_id','pair']
M=pd.concat([qex,qtr,qqu,qwi],axis=1)
rng=np.random.default_rng(0)
def ci(g):
    r=g.groupby('race_id').agg(s=('ret','size'),r=('ret','sum'))
    if len(r)<5: return f"R={len(r)}"
    s,v=r.s.values*100.,r.r.values.astype(float); idx=rng.integers(0,len(r),(400,len(r))); b=v[idx].sum(1)/s[idx].sum(1)*100
    return f"{v.sum()/s.sum()*100:6.1f}% [{np.percentile(b,2.5):3.0f}-{np.percentile(b,97.5):3.0f}] R={len(r)} 点={int(s.sum()/100)}"
out={}
for bt in ['2車複','ワイド','2車単']:
    x=d[d.bt==bt].copy(); x['pair']=x.combo.map(pair)
    x=x.join(M,on=['race_id','pair'])
    if bt=='2車単':
        rev=x.set_index(['race_id','combo']).q
        x['q_rev']=[rev.get((r,f"{c.split('>')[1]}>{c.split('>')[0]}"),np.nan) for r,c in zip(x.race_id,x.combo)]
        x['share']=x.q/(x.q+x.q_rev)
    cols=['q','q_ex','q_tr','q_qu','q_wi']+(['share'] if bt=='2車単' else [])
    x=x.dropna(subset=cols)
    F=np.column_stack([np.log(x[c].clip(1e-6)) for c in cols]+[np.log(x.odds),(x.nr==9).astype(float)])
    y=(x.ret>0).astype(int).values; tr_=(x.date<CUT).values
    lr=LogisticRegression(C=1.0,max_iter=2000).fit(F[tr_],y[tr_])
    x['pm']=lr.predict_proba(F)[:,1]; x['ev']=x.pm*x.odds
    te=x[~tr_]
    print(f"== {bt}  係数", dict(zip(cols+['log_odds','9車'],np.round(lr.coef_[0],2))))
    print("  テスト期間 当たり具合 実際/予想:", round(te.ret.gt(0).sum()/te.pm.sum(),3))
    for th in [0.9,1.0,1.05,1.1,1.2]:
        for mx in [30,100]:
            g=te[(te.ev>=th)&(te.odds<=mx)]; print(f"  EV≥{th} {mx}倍以下: ",ci(g))
    out[bt]={'cols':cols,'mu':lr.mu,'sd':lr.sd,'w':lr.w}
pd.to_pickle(out,f'cross2_model_{CUT}.pkl')
