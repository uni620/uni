"""頭AI（確率）＋相手AI（2着・3着もAI上位）を 4月〜11月の8か月で検証（各月その月より前だけで学習）。"""
import hx_lib
from hx_lib import *
import sqlite3, itertools
M=[('2026-04-01','2026-05-01'),('2026-05-01','2026-06-01'),('2026-06-01','2026-07-01'),('2026-07-01','2026-08-01'),('2026-08-01','2026-09-01'),
   ('2026-09-01','2026-10-01'),('2026-10-01','2026-11-01'),('2026-11-01','2026-11-21')]
E=pd.read_pickle(f'{S}/hx2.pkl'); C=pickle.load(open(f'{S}/hx_cols.pkl','rb'))
PFC,CC=C['PFC'],C['CC']; FEAT=[f for f in C['FEAT'] if f not in PFC]; base=FEAT+CC
exec(open(f'{S}/hx_2nd.py').read().split("R2=")[0].split("H=pd.read_pickle")[1].split('\n',1)[1])   # rel() を読み込む
R2=['頭と同じライン','頭の位置','頭の直後','頭の直前','頭が先頭','頭が番手','頭が単騎','頭のライン人数','頭との得点差','残りでの得点順位','残りでの勝率順位']
F2=base+R2
PB=dict(objective='binary',learning_rate=0.03,num_leaves=15,min_data_in_leaf=200,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,verbose=-1)
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
pay=pd.read_sql("select race_id,bet_type,combo,payout from payouts",c); PAY={(r,b,k):v for r,b,k,v in pay.itertuples(index=False)}
rows=[]
for a,b in M:
    x=E[E.date<a]; y=E[(E.date>=a)&(E.date<b)].copy()
    p=sum(lgb.train(dict(PB,seed=s),lgb.Dataset(x[base],x.win),500).predict(y[base]) for s in (1,2))
    y['pr']=p; y['pr']/=y.groupby('race_id').pr.transform('sum')
    hh=y.loc[y.groupby('race_id').pr.idxmax()].set_index('race_id')
    hw=x[x.finish==1].set_index('race_id').car_no
    X=rel(x,hw).sort_values(['race_id','car_no']); lab=np.select([X.finish==2,X.finish==3],[2,1],0)
    d=lgb.Dataset(X[F2],lab); d.set_group(X.groupby('race_id',sort=False).size().values)
    ms=[lgb.train(dict(BASEP,seed=s),d,400) for s in (1,2)]
    Y=rel(y,hh.car_no).sort_values(['race_id','car_no'])
    Y['s']=sum(pd.Series(m.predict(Y[F2]),index=Y.index).groupby(Y.race_id).rank() for m in ms)+1e-6*np.arange(len(Y))
    opp=Y.sort_values(['race_id','s'],ascending=[True,False]).groupby('race_id').car_no.apply(list)
    for rid,h in hh.iterrows():
        if rid not in opp.index: continue
        o=opp[rid]; hd=int(h.car_no)
        P={'2車単 1点':[('2車単',f'{hd}>{o[0]}')],'2車単 3点':[('2車単',f'{hd}>{z}') for z in o[:3]],
           '3連複 3点':[('3連複','-'.join(map(str,sorted((hd,u,v))))) for u,v in itertools.combinations(o[:3],2)],
           '3連単 6点':[('3連単',f'{hd}>{u}>{v}') for u in o[:2] for v in o[:4] if u!=v],
           '3連単 9点':[('3連単',f'{hd}>{u}>{v}') for u in o[:3] for v in o[:4] if u!=v]}
        for nm,bt in P.items():
            rows.append((a[:7],rid,h.pr,h.win,nm,len(bt)*100,sum(PAY.get((rid,t,k),0) for t,k in bt)))
    print(a[:7],'done',flush=True)
R=pd.DataFrame(rows,columns=['月','race_id','pr','headwin','買い方','cost','ret']); R.to_pickle(f'{S}/hx_verify.pkl')
