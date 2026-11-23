import pandas as pd, numpy as np
from scipy.optimize import minimize
CUT='2026-10-15'
x=pd.read_pickle('line_2車複.pkl')
x=x[x.groupby('race_id').rel.transform(lambda s:(s!='不明').all())].copy()
x['win']=(x.ret>0).astype(float)
R=['同ライン先頭-番手','同ライン番手-3番手','同ラインその他']
X=np.column_stack([np.log(x.p.clip(1e-9))]+[(x.rel==r).astype(float) for r in R]+[np.log(x.q.clip(1e-9))])
gid=x.groupby('race_id').ngroup().values; win=x.win.values
tr=(x.date<CUT).values; te=~tr
def nll(w,idx,cols):
    z=X[idx][:,cols]@w; g=gid[idx]
    s=pd.Series(z); mz=s.groupby(g).transform('max').values
    lse=np.log(pd.Series(np.exp(z-mz)).groupby(g).transform('sum').values)+mz
    return -((z-lse)*win[idx]).sum()/win[idx].sum()
res={}
for name,cols in [('AIのみ(今)',[0]),('AI+ライン',[0,1,2,3]),('市場のみ',[4]),('AI+市場',[0,4]),('AI+ライン+市場',[0,1,2,3,4])]:
    w0=np.ones(len(cols))*0.5
    w=minimize(lambda w:nll(w,tr,cols),w0,method='Nelder-Mead',options={'maxiter':4000}).x
    res[name]=(np.round(w,3),round(nll(w,te,cols),4))
    print(name,'重み',np.round(w,3),'テスト対数損失',round(nll(w,te,cols),4),flush=True)
    if name=='AI+ライン+市場':
        z=X[:,cols]@w; s=pd.Series(z); pb=np.exp(z-s.groupby(gid).transform('max').values); pb/=pd.Series(pb).groupby(gid).transform('sum').values
        x['pb']=pb; x['evb']=pb*x.odds
    if name=='AI+ライン':
        z=X[:,cols]@w; s=pd.Series(z); pl=np.exp(z-s.groupby(gid).transform('max').values); pl/=pd.Series(pl).groupby(gid).transform('sum').values
        x['pl']=pl; x['evl']=pl*x.odds
y=x[te]
rng=np.random.default_rng(0)
def ci(g):
    r=g.groupby('race_id').agg(s=('ret','size'),r=('ret','sum'))
    if len(r)<5: return f"R={len(r)}"
    s,v=r.s.values*100.,r.r.values.astype(float); idx=rng.integers(0,len(r),(400,len(r))); b=v[idx].sum(1)/s[idx].sum(1)*100
    return f"{v.sum()/s.sum()*100:.1f}% [{np.percentile(b,2.5):.0f}-{np.percentile(b,97.5):.0f}] R={len(r)} 点={int(s.sum()/100)}"
print('\nテスト期間(10/15-11/19) 2車複')
print('今のAI本命1点     ', ci(y[y.groupby('race_id').p.rank(ascending=False,method='first')==1]))
print('ライン補正AI本命1点', ci(y[y.groupby('race_id').pl.rank(ascending=False,method='first')==1]))
print('1番人気1点        ', ci(y[y.rk_q==1]))
for th in [1.0,1.1,1.2,1.5]:
    print(f'ライン補正AI EV≥{th}', ci(y[(y.evl>=th)&(y.odds<=50)]))
for th in [0.9,1.0,1.05,1.1]:
    print(f'混合 EV≥{th}', ci(y[(y.evb>=th)&(y.odds<=50)]))
x.to_pickle('line2_out.pkl')
