"""買い目の見込み確率の補正（オッズは使わない）。月ごとに、その月より前の結果だけで補正式を作る（ロジスティック: logit(P), logit(頭の確率)）。"""
import pandas as pd,numpy as np
from scipy.optimize import minimize
class LogisticRegression:
    def __init__(self,C=10): self.C=C
    def fit(self,X,y):
        X=np.column_stack([np.ones(len(X)),np.asarray(X,float)]); y=np.asarray(y,float)
        def f(w):
            z=X@w; p=1/(1+np.exp(-z)); l=-(y*np.log(p+1e-12)+(1-y)*np.log(1-p+1e-12)).sum()+0.5/self.C*(w[1:]**2).sum()
            g=X.T@(p-y); g[1:]+=w[1:]/self.C; return l,g
        self.w=minimize(f,np.zeros(X.shape[1]),jac=True,method='L-BFGS-B').x; return self
    def predict_proba(self,X):
        X=np.column_stack([np.ones(len(X)),np.asarray(X,float)]); p=1/(1+np.exp(-(X@self.w))); return np.column_stack([1-p,p])
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
B=pd.read_pickle(f'{S}/hx_cut.pkl'); B['hit']=(B.pay>0).astype(int)
lg=lambda p:np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
B['x1']=lg(B.P); B['x2']=lg(B.pr)
months=sorted(B.月.unique()); B['Pc']=np.nan
for m in months[1:]:
    for bt in ('2車単','3連単'):
        tr=B[(B.月<m)&(B.bt==bt)]; te=(B.月==m)&(B.bt==bt)
        lr=LogisticRegression(C=10).fit(tr[['x1','x2']],tr.hit); B.loc[te,'Pc']=lr.predict_proba(B.loc[te,['x1','x2']])[:,1]
X=B[B.Pc.notna()&(B.pr>=0.6)].copy()
print('確認期間',X.月.min(),'〜',X.月.max(),'自信60%以上',X.race_id.nunique(),'R')
for bt in ('2車単','3連単'):
    x=X[X.bt==bt].copy(); x['帯']=pd.qcut(x.Pc,5)
    t=x.groupby('帯',observed=True).agg(補正前=('P','mean'),補正後=('Pc','mean'),実際=('hit','mean'),点=('hit','size'))
    print('---',bt,'（%）'); print((t[['補正前','補正後','実際']]*100).round(1).join(t.点).to_string())
X['EVc']=X.Pc*X.odds; X['EV']=X.P*X.odds
rng=np.random.default_rng(0)
def summ(x,lab):
    if len(x)==0: return dict(切り方=lab)
    g=x.groupby('race_id').agg(c=('pay','size'),r=('pay','sum'))
    s=[g.iloc[rng.integers(0,len(g),len(g))].pipe(lambda d:d.r.sum()/d.c.sum()) for _ in range(1000)]
    mo=x.groupby('月').apply(lambda d:int(round(d.pay.sum()/len(d)))).to_dict()
    return dict(切り方=lab,点=len(x),レース=x.race_id.nunique(),当たったレース率=round(x[x.pay>0].race_id.nunique()/x.race_id.nunique()*100,1),回収率=round(x.pay.sum()/len(x),1),
                幅=f"{np.percentile(s,2.5):.0f}〜{np.percentile(s,97.5):.0f}",月別=mo)
for bt in ('2車単','3連単'):
    x=X[X.bt==bt]; out=[summ(x,'切らない')]
    for t in (1.0,1.2,1.5):
        out.append(summ(x[x.EV>t],f'補正前EV{t}超')); out.append(summ(x[x.EVc>t],f'補正後EV{t}超'))
    print('===',bt); print(pd.DataFrame(out).to_string(index=False))
B.to_pickle(f'{S}/hx_calib.pkl')
