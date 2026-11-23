import pickle, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
def load():
    E=pd.read_pickle(f'{S}/hx.pkl'); C=pickle.load(open(f'{S}/hx_cols.pkl','rb'))
    E=E[E.groupby('race_id').win.transform('sum')==1].sort_values(['date','race_id','car_no']).reset_index(drop=True)
    return E,C
MONTHS=[('2026-09-01','2026-10-01'),('2026-10-01','2026-11-01'),('2026-11-01','2026-11-21')]
BASEP=dict(objective='lambdarank',learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,verbose=-1,eval_at=[1],lambdarank_truncation_level=3)
def label(x,kind):
    if kind=='rank3': return np.select([x.finish==1,x.finish==2,x.finish==3],[3,2,1],0)
    if kind=='win': return (x.finish==1).astype(int).values
    if kind=='rank5': return np.select([x.finish==1,x.finish==2,x.finish==3,x.finish==4],[4,3,2,1],0)
def evaluate(E,F,params=None,rounds=400,seeds=(1,2),lab='rank3',train_from=None,ret=False):
    P=dict(BASEP,**(params or {}))
    hits=[];n=[];preds=[]
    for a,b in MONTHS:
        trm=(E.date<a)&((E.date>=train_from) if train_from else True); tem=(E.date>=a)&(E.date<b)
        x=E[trm]; y=E[tem]; ps=np.zeros(len(y))
        for sd in seeds:
            if P['objective']=='binary':
                pp={k:v for k,v in P.items() if k not in('eval_at','lambdarank_truncation_level')}
                d=lgb.Dataset(x[F],(x.finish==1).astype(int)); m=lgb.train(dict(pp,seed=sd),d,rounds)
            else:
                d=lgb.Dataset(x[F],label(x,lab)); d.set_group(x.groupby('race_id',sort=False).size().values); m=lgb.train(dict(P,seed=sd),d,rounds)
            p=m.predict(y[F]); ps+=pd.Series(p,index=y.index).groupby(y.race_id).rank().values
        yy=y.assign(p=ps+1e-6*np.arange(len(y))); t=yy.loc[yy.groupby('race_id').p.idxmax()]
        hits.append(t.win.sum()); n.append(len(t)); preds.append(yy[['race_id','car_no','p']])
    r=[h/k*100 for h,k in zip(hits,n)]; tot=sum(hits)/sum(n)*100
    out=(round(tot,2),[round(v,1) for v in r],sum(n))
    return (out,pd.concat(preds),m) if ret else out
