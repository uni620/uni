from hx_lib import *
E=pd.read_pickle(f'{S}/hx2.pkl'); C=pickle.load(open(f'{S}/hx_cols.pkl','rb'))
PFC,CC=C['PFC'],C['CC']; FEAT=[f for f in C['FEAT'] if f not in PFC]; base=FEAT+CC
# 学習データの量: 各月の直前 N か月だけで学習
import datetime as dt
for months in (2,4,6):
    hits=n=0; res=[]
    for a,b in MONTHS:
        st=(pd.Timestamp(a)-pd.DateOffset(months=months)).strftime('%Y-%m-%d')
        P=dict(BASEP); x=E[(E.date>=st)&(E.date<a)]; y=E[(E.date>=a)&(E.date<b)]; ps=np.zeros(len(y))
        for sd in (1,2):
            d=lgb.Dataset(x[base],label(x,'rank3')); d.set_group(x.groupby('race_id',sort=False).size().values)
            m=lgb.train(dict(P,seed=sd),d,400); ps+=pd.Series(m.predict(y[base]),index=y.index).groupby(y.race_id).rank().values
        yy=y.assign(p=ps+1e-6*np.arange(len(y))); t=yy.loc[yy.groupby('race_id').p.idxmax()]; hits+=t.win.sum(); n+=len(t); res.append(round(t.win.mean()*100,1))
    print(f'直前{months}か月で学習',round(hits/n*100,2),res,flush=True)
print('全期間(1月〜)で学習',evaluate(E,base),flush=True)
