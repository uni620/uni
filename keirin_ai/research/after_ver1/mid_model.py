"""中穴（3連単20〜100倍）の目の中で、来る目を当てるモデル（人気＋目の形＋選手・レースの材料）。月ごとに学習し直し 8〜11月で評価。"""
import sqlite3,pandas as pd,numpy as np,lightgbm as lgb,warnings
warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
AR=pd.read_pickle(f'{S}/ana_R.pkl').set_index('race_id')
od=pd.read_sql("select race_id,combo,odds from odds where is_final=1 and bet_type='3連単' and odds>0",c).drop_duplicates(['race_id','combo'])
od=od[od.race_id.isin(AR.index)]
pay=pd.read_sql("select race_id,combo,payout from payouts where bet_type='3連単'",c); PAY=dict(zip(zip(pay.race_id,pay.combo),pay.payout))
od['q']=1/od.odds; od['q']=od.q/od.groupby('race_id').q.transform('sum')
ex=pd.read_sql("select race_id,combo,odds from odds where is_final=1 and bet_type='2車単' and odds>0",c).drop_duplicates(['race_id','combo'])
ex['car']=ex.combo.str.split('>').str[0].astype(int); ex['iq']=1/ex.odds
W=ex.groupby(['race_id','car']).iq.sum().reset_index(); W['人気']=W.groupby('race_id').iq.rank(ascending=False,method='first'); W['pw']=W.iq/W.groupby('race_id').iq.transform('sum')
POP=W.set_index(['race_id','car'])
E=pd.read_pickle(f'{S}/hx2.pkl')[['race_id','car_no','lkey','line_pos','先頭','番手','単騎','得点順位','pB','自分のラインのpB','B率','逃率','差率','近5_fin_pct']]
K=E.set_index(['race_id','car_no'])
D=pd.read_pickle(f'{S}/breakfeat.pkl')
B=od[(od.odds>=20)&(od.odds<=100)].copy()
B['帯内順位']=B.groupby('race_id').odds.rank(method='first'); B['帯の目数']=B.groupby('race_id').combo.transform('size'); B['帯の人気合計']=B.groupby('race_id').q.transform('sum')
cs=B.combo.str.split('>',expand=True).astype(int)
for i in range(3):
    idx=pd.MultiIndex.from_arrays([B.race_id,cs[i]])
    B[f'人気{i+1}']=POP['人気'].reindex(idx).values; B[f'1着確率{i+1}']=POP['pw'].reindex(idx).values
    for col in ['lkey','line_pos','先頭','番手','単騎','得点順位','pB','自分のラインのpB','B率','差率','近5_fin_pct']:
        B[f'{col}{i+1}']=K[col].reindex(idx).values
B['同12']=(B.lkey1==B.lkey2).astype(float); B['同13']=(B.lkey1==B.lkey3).astype(float); B['同23']=(B.lkey2==B.lkey3).astype(float)
B['直後12']=(B.同12==1)&(B.line_pos2==B.line_pos1+1)
B=B.join(D.drop(columns=['崩れ','20倍以上','date'],errors='ignore'),on='race_id')
B['date']=B.race_id.map(AR.date); B['pay']=[PAY.get(k,0) for k in zip(B.race_id,B.combo)]; B['hit']=(B.pay>0).astype(int)
F=[c for c in B.columns if c not in ('race_id','combo','odds','pay','hit','date','lkey1','lkey2','lkey3')]
B=B.sort_values(['race_id','odds']).reset_index(drop=True)
print('帯の目',len(B),'レース',B.race_id.nunique(),flush=True)
P=dict(objective='binary',learning_rate=0.03,num_leaves=31,min_data_in_leaf=300,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,verbose=-1,seed=1)
out=[]
for a,b in (('2026-08-01','2026-09-01'),('2026-09-01','2026-10-01'),('2026-10-01','2026-11-01'),('2026-11-01','2026-11-21')):
    tr=B[B.date<a]; te=B[(B.date>=a)&(B.date<b)].copy()
    te['pm']=sum(lgb.train(dict(P,seed=s),lgb.Dataset(tr[F],tr.hit),500).predict(te[F]) for s in (1,2))/2
    out.append(te[['race_id','date','combo','odds','q','pay','pm']]); print(a[:7],'done',flush=True)
Y=pd.concat(out); Y.to_pickle(f'{S}/mid_model_Y.pkl')
for lab,col in (('人気順','q'),('中穴モデル','pm')):
    Y=Y.sort_values(['race_id',col],ascending=[True,False]); Y['rk']=Y.groupby('race_id').cumcount()+1
    for k in (3,5,10,15):
        g=Y[Y.rk<=k].groupby('race_id').pay.agg(['size','sum']); m=Y[Y.rk<=k].groupby(Y[Y.rk<=k].date.str[:7]).apply(lambda d:round((d.groupby('race_id').pay.sum()>0).mean()*100,1)).tolist()
        print(f'{lab} {k}点: 的中 {(g["sum"]>0).mean()*100:.1f}%  回収 {g["sum"].sum()/g["size"].sum():.1f}%  月別的中{m}')
