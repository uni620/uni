"""頭AI 実験用の土台（全特徴をキャッシュ）。元: 頭AI v1（オッズを使わず1着を当てる）。元: 展開AI v3（方針: オッズに反映されにくい要素だけを学ぶ）。
- 出発点 = 人気（2車単の確定オッズから出した1着の確率）。AIは「人気からのずれ」だけを学ぶ（init_score）。
- 特徴: 選手の得意な展開（B取り時・先頭・番手・単騎での成績）、本当の調子（直近5走の上がり順位・勝者からの着差）、
  展開（主導権の確率 pB と、ラインの主導権・他ラインの最大）。すべてそのレースより前の成績だけで計算。
学習 6/11〜8/31、回数決め 9/1〜9/30、確認 10/1〜11/20（1回だけ）。"""
import sys, sqlite3, itertools, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/home/user/uni/keirin_ai')
from keirin.tenkai import features, stage2_features, QUERY
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite',timeout=60)
q=QUERY.replace("r.date, r.race_class, i.sb","r.date, r.race_class, i.sb, i.agari, i.margin, e.rider_id, e.score, r.venue, r.race_name, r.grade, r.bank_len, r.race_no, e.win_rate, e.ren2, e.ren3")
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

T2=pd.read_pickle(f'{S}/tenkai2.pkl')[['race_id','car_no','ちぎられ率','ちぎり率','まくり率','まくり回数','上がり差','B時の平均着順']]
E=E.merge(T2,on=['race_id','car_no'],how='left')
for k in ('score','win_rate','ren2','ren3'): E[k]=pd.to_numeric(E[k],errors='coerce')
E['得点差']=E.score-E.groupby('race_id').score.transform('mean'); E['得点順位']=E.groupby('race_id').score.rank(ascending=False)
E['得点1位との差']=E.score-E.groupby('race_id').score.transform('max')
E['自ライン平均得点']=E.groupby('lk').score.transform('mean'); E['自ライン得点順位']=E.groupby('race_id').自ライン平均得点.rank(ascending=False,method='dense')
E['勝率順位']=E.groupby('race_id').win_rate.rank(ascending=False)
from keirin.config import load_config
from keirin.db import DB
from keirin import pipeline
from keirin.features import FEATURES as PF
cfg=load_config('/home/user/uni/keirin_ai/config.toml')
F=pipeline.features_for(DB('/home/user/uni/keirin_ai/data/keirin.sqlite'),cfg)
PF2=[x for x in PF if x not in ('pos_type',)]+['age','gear_n']
F['gear_n']=pd.to_numeric(F.get('gear'),errors='coerce'); F['age']=pd.to_numeric(F.get('age'),errors='coerce')
F=F[['race_id','car_no']+[x for x in PF2 if x in F]].rename(columns={x:'pf_'+x for x in PF2})
E=E.merge(F,on=['race_id','car_no'],how='left')
PFC=[c for c in E.columns if c.startswith('pf_')]
for k in ('pf_win_3m','pf_top3_3m','pf_recent_finish','pf_trend','pf_kaku_member_diff'):
    if k in E: E[k+'_順位']=E.groupby('race_id')[k].rank(ascending=(k=='pf_recent_finish')); PFC.append(k+'_順位')
print('追加した特徴',len(PFC),flush=True)
E=E[E.groupby('race_id').win.transform('sum')==1]
FEAT=['B_1着率','B_3着内率','B_回数','先_1着率','先_3着内率','先_回数','番_1着率','番_3着内率','番_回数','単_3着内率',
      '近5_agari_pct','近5_behind','近5_fin_pct','内容−着順','今回の位置の3着内率',
      'pB','自分のラインのpB','自分の先頭のpB','他ラインの最大pB','競り合い','主導権×B時1着率',
      '先頭','番手','3番手+','単騎','ライン人数','n','ライン数','先行候補数','逃率','捲率','差率','マ率','B率','S率',
      'ちぎられ率','ちぎり率','まくり率','まくり回数','上がり差','B時の平均着順',
      'score','得点差','得点順位','得点1位との差','自ライン平均得点','自ライン得点順位','win_rate','ren2','ren3','勝率順位']+PFC

X3=pd.read_pickle(f'{S}/head3_feats.pkl'); E=E.merge(X3,on=['race_id','car_no'],how='left')
E['FEAT_BASE']=0
import pickle
pickle.dump({'FEAT':FEAT,'PFC':PFC,'CM':[c for c in X3.columns if c.startswith(('cm_','R_','先頭の'))],
  'CC':['連係回数','連係時の先頭3着内率','連係時のワンツー率','対戦数','相手への先着率','先着率順位','番手との連係回数']},open(f'{S}/hx_cols.pkl','wb'))
E.to_pickle(f'{S}/hx.pkl'); print(E.shape, E.race_id.nunique())
