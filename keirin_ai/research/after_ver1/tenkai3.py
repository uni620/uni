"""展開AI v3（方針: オッズに反映されにくい要素だけを学ぶ）。
- 出発点 = 人気（2車単の確定オッズから出した1着の確率）。AIは「人気からのずれ」だけを学ぶ（init_score）。
- 特徴: 選手の得意な展開（B取り時・先頭・番手・単騎での成績）、本当の調子（直近5走の上がり順位・勝者からの着差）、
  展開（主導権の確率 pB と、ラインの主導権・他ラインの最大）。すべてそのレースより前の成績だけで計算。
学習 6/11〜8/31、回数決め 9/1〜9/30、確認 10/1〜11/20（1回だけ）。"""
import sys, sqlite3, itertools, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/home/user/uni/keirin_ai')
from keirin.tenkai import features, stage2_features, QUERY
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite',timeout=60)
q=QUERY.replace("r.date, r.race_class, i.sb","r.date, r.race_class, i.sb, i.agari, i.margin, e.rider_id")
E=features(pd.read_sql(q,c)); E['finish']=pd.to_numeric(E.finish,errors='coerce')
scr=set(E[E.scratched==1].race_id); E=E[~E.race_id.isin(scr)].copy()
E['dt']=pd.to_datetime(E.date)
# 着差 → 勝者からの差（車身）
M={'同着':0,'微差':.05,'タイヤ':.05,'1/8車輪':.06,'1/4車輪':.1,'1/2車輪':.2,'3/4車輪':.3,'1車輪':.4,'1/2車身':.5,'3/4車身':.75,'1車身':1,'1車身1/2':1.5,'大差':10}
def mv(s):
    if s is None or (isinstance(s,float) and np.isnan(s)): return 0.0
    s=str(s)
    if s in M: return M[s]
    if s.endswith('車身') and s[:-2].isdigit(): return float(s[:-2])
    return np.nan
E['m1']=E.margin.map(mv)
E=E.sort_values(['race_id','finish'])
E['behind']=E.groupby('race_id').m1.cumsum().clip(upper=10)
E.loc[E.finish.isna(),'behind']=np.nan
E['agari']=pd.to_numeric(E.agari,errors='coerce')
E['agari_pct']=E.groupby('race_id').agari.rank(pct=True)          # 小さいほど速い
E['fin_pct']=E.finish/E.n
E['gotB']=E.sb.astype(str).str.contains('B').astype(float)
E['top3']=(E.finish<=3).astype(float); E['win']=(E.finish==1).astype(float)
# 状況別の出来事（その日の合計→選手ごとに過去120日、当日は含めない）
for k,cond in (('B',E.gotB==1),('先',E.先頭==1),('番',E.番手==1),('単',E.単騎==1)):
    E[f'n_{k}']=cond.astype(float); E[f'w_{k}']=(cond&(E.win==1)).astype(float); E[f't_{k}']=(cond&(E.top3==1)).astype(float)
ev=[f'{a}_{k}' for k in 'B先番単' for a in 'nwt']
g=E.groupby(['rider_id','dt'])[ev].sum().reset_index().sort_values(['rider_id','dt'])
parts=[]
for rid,x in g.groupby('rider_id'):
    y=x.set_index('dt')[ev].rolling('120D',closed='left').sum(); y['rider_id']=rid; parts.append(y.reset_index())
R=pd.concat(parts); E=E.merge(R,on=['rider_id','dt'],how='left',suffixes=('','_h'))
for k in 'B先番単':
    n=E[f'n_{k}_h']; E[f'{k}_1着率']=(E[f'w_{k}_h']+0.1)/(n+1); E[f'{k}_3着内率']=(E[f't_{k}_h']+0.33)/(n+1); E[f'{k}_回数']=n
# 本当の調子: 直近5走（当日より前）の上がり順位・勝者からの差・着順、とその「内容−着順」
E=E.sort_values(['rider_id','dt','race_id'])
for col in ('agari_pct','behind','fin_pct'):
    s=E.groupby('rider_id')[col].shift(1)   # 同じ日の前のレースも入るが、当日の先のレースではない
    E[f'近5_{col}']=s.groupby(E.rider_id).transform(lambda v:v.rolling(5,min_periods=2).mean())
E['内容−着順']=E['近5_fin_pct']-E['近5_agari_pct']     # プラス=着順より内容が良い
# 状況に合わせた「今回の位置での得意度」
E['今回の位置の3着内率']=np.select([E.先頭==1,E.番手==1,E.単騎==1],[E['先_3着内率'],E['番_3着内率'],E['単_3着内率']],np.nan)
# 展開: 段階1の pB（v1 で学習外に作ったもの）
T1=pd.read_pickle(f'{S}/tenkai.pkl')[['race_id','car_no','pB']]
E=E.merge(T1,on=['race_id','car_no'],how='left'); E['pB']=E.pB.fillna(E.pB.mean())
E=stage2_features(E)
E['主導権×B時1着率']=E.pB*E['B_1着率']
# 人気（2車単の確定オッズ → 1着の確率）
od=pd.read_sql("select race_id,combo,odds from odds where is_final=1 and bet_type='2車単' and odds>0",c).drop_duplicates(['race_id','combo'])
od['car_no']=od.combo.str.split('>').str[0].astype(int); od['iq']=1/od.odds
Q=od.groupby(['race_id','car_no']).iq.sum().reset_index(); Q['q']=Q.iq/Q.groupby('race_id').iq.transform('sum')
E=E.merge(Q[['race_id','car_no','q']],on=['race_id','car_no'])
E=E[E.groupby('race_id').win.transform('sum')==1]
E=E[E.groupby('race_id').car_no.transform('size')==E.n]
E['lq']=np.log(E.q.clip(1e-4)); E['logit_q']=np.log(E.q.clip(1e-4)/(1-E.q.clip(upper=.999)))
E['人気順']=E.groupby('race_id').q.rank(ascending=False)
FEAT=['B_1着率','B_3着内率','B_回数','先_1着率','先_3着内率','先_回数','番_1着率','番_3着内率','番_回数','単_3着内率',
      '近5_agari_pct','近5_behind','近5_fin_pct','内容−着順','今回の位置の3着内率',
      'pB','自分のラインのpB','自分の先頭のpB','他ラインの最大pB','競り合い','主導権×B時1着率',
      '先頭','番手','3番手+','単騎','ライン人数','n','ライン数','先行候補数','lq','人気順']
E=E.sort_values(['date','race_id','car_no']).reset_index(drop=True)
tr=(E.date<'2026-09-01'); va=(E.date>='2026-09-01')&(E.date<'2026-10-01'); te=E.date>='2026-10-01'
P=dict(objective='binary',learning_rate=0.02,num_leaves=7,min_data_in_leaf=200,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,verbose=-1,seed=1)
def race_ll(p,df):
    p=p/ pd.Series(p,index=df.index).groupby(df.race_id).transform('sum').values
    return -np.log(np.clip(p[df.win.values==1],1e-9,1))
# 人気の基準: q^a（a は学習期間で決める）
from scipy.optimize import minimize_scalar
def qa(a,df): v=df.q.values**a; return v
a=minimize_scalar(lambda a:race_ll(qa(a,E[tr]),E[tr]).mean(),bounds=(0.5,2),method='bounded').x
print('人気の補正 a=',round(a,3))
E['base']=qa(a,E); E['base']=E.base/E.groupby('race_id').base.transform('sum')
init=np.log(E.base.clip(1e-5)/(1-E.base.clip(upper=.999)))
dtr=lgb.Dataset(E.loc[tr,FEAT],E.loc[tr,'win'],init_score=init[tr]); dva=lgb.Dataset(E.loc[va,FEAT],E.loc[va,'win'],init_score=init[va])
ll=[]
m=lgb.train(P,dtr,600,valid_sets=[dva],callbacks=[lgb.record_evaluation(r:={}),])
curve=r['valid_0']['binary_logloss']; best=int(np.argmin(curve))+1; print('回数',best,'(9月の検証で決定)')
# 学習+9月で学び直して 10/1〜 を予想
tv=tr|va
m=lgb.train(P,lgb.Dataset(E.loc[tv,FEAT],E.loc[tv,'win'],init_score=init[tv]),best)
raw=m.predict(E[FEAT],raw_score=True)+init
E['p3']=1/(1+np.exp(-raw)); E['p3']=E.p3/E.groupby('race_id').p3.transform('sum')
X=E[te]
d=race_ll(X.base.values,X)-race_ll(X.p3.values,X)
rng=np.random.default_rng(0); bs=[d[rng.integers(0,len(d),len(d))].mean()*1000 for _ in range(2000)]
print(f'確認期間 {X.race_id.nunique()}R: 人気からの上乗せ {d.mean()*1000:+.2f}‰ [{np.percentile(bs,2.5):+.2f}〜{np.percentile(bs,97.5):+.2f}]')
for col,lab in (('base','人気'),('p3','展開AI v3')):
    t=X.loc[X.groupby('race_id')[col].idxmax()]; print(lab,'本命の1着率',round(t.win.mean()*100,1),'%')
imp=pd.Series(m.feature_importance('gain'),index=FEAT); print('効いた特徴',(imp/imp.sum()).sort_values(ascending=False).head(12).round(3).to_dict())
E[['race_id','date','car_no','win','finish','q','base','p3']].to_pickle(f'{S}/tenkai3.pkl')
