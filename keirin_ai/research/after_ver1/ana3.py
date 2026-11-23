import pandas as pd, numpy as np
from scipy.optimize import minimize
d=pd.read_pickle('odds_val.pkl'); pd.set_option('display.width',250)
rng=np.random.default_rng(0)
def ci(g):
    r=g.groupby('race_id').agg(s=('ret','size'),r=('ret','sum'))
    if len(r)<5: return (len(r),np.nan,'-')
    s,v=r.s.values*100.,r.r.values.astype(float); idx=rng.integers(0,len(r),(400,len(r))); b=v[idx].sum(1)/s[idx].sum(1)*100
    return (len(r),round(v.sum()/s.sum()*100,1),f"{np.percentile(b,2.5):.0f}-{np.percentile(b,97.5):.0f}")
CUT='2026-10-15'
out=[];fit=[]
for (m,bt),x in d[d.bt!='ワイド'].groupby(['method','bt']):
    x=x.copy(); x['win']=(x.ret>0).astype(float); x['lp']=np.log(x.p.clip(1e-6)); x['lq']=np.log(x.q.clip(1e-6))
    gid=x.groupby('race_id').ngroup().values
    def nll(w,idx):
        z=w[0]*x.lp.values[idx]+w[1]*x.lq.values[idx]; g=gid[idx]
        mz=pd.Series(z).groupby(g).transform('max').values; e=np.exp(z-mz)
        lse=np.log(pd.Series(e).groupby(g).transform('sum').values)+mz
        return -((z-lse)*x.win.values[idx]).sum()/x.win.values[idx].sum()
    tr=(x.date<CUT).values; te=~tr
    w=minimize(lambda w:nll(w,tr),[0.5,0.5],method='Nelder-Mead').x
    fit.append([m,bt,round(w[0],3),round(w[1],3),round(nll([1,0],te),4),round(nll([0,1],te),4),round(nll(w,te),4)])
    z=w[0]*x.lp+w[1]*x.lq; x['pb']=np.exp(z-z.groupby(x.race_id).transform('max')); x['pb']/=x.pb.groupby(x.race_id).transform('sum')
    x['evb']=x.pb*x.odds
    y=x[te]
    for th in [1.0,1.1,1.2,1.5]:
        a=ci(y[(y.evb>=th)&(y.odds<=50)]); out.append([m,bt,th,*a])
    a=ci(y[y.groupby('race_id').pb.rank(ascending=False,method='first')==1]); out.append([m,bt,'混合本命',*a])
print("[F] 混合モデル  確率 ∝ AI^a × 市場^b （9/09-10/14で学習）。テスト(10/15-11/19)の対数損失：小さいほど良い")
print(pd.DataFrame(fit,columns=['方法','券種','a(AI)','b(市場)','AIのみ','市場のみ','混合']).to_string())
print("\n[G] テスト期間：混合EV≥t を全部買う（オッズ50倍以下）")
print(pd.DataFrame(out,columns=['方法','券種','EV≥','R','回収%','95%幅']).to_string())
