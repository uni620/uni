"""展開AI v2: v1 ＋ 強さ（競走得点）・スタミナ（上がり、Bを取ったときの着順）・直近90日の ちぎられ率/ちぎり率/まくり。
学習 1/1〜8/31、確認 9/1〜11/21（v1 と同じ）。"""
import sys, sqlite3, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/home/user/uni/keirin_ai')
from keirin.tenkai import features, stage2_features, A, B, P, QUERY
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite',timeout=60)
q=QUERY.replace("r.date, r.race_class, i.sb","r.date, r.race_class, i.sb, i.agari, e.rider_id, e.score, e.kimarite")
E=features(pd.read_sql(q,c)); E['finish']=pd.to_numeric(E.finish,errors='coerce'); E['score']=pd.to_numeric(E.score,errors='coerce')
scr=set(E[E.scratched==1].race_id); E=E[~E.race_id.isin(scr)].copy()
E['dt']=pd.to_datetime(E.date)
# --- そのレースでの出来事（あとで90日分を数える）
ld=E[E.先頭==1][['race_id','lk','finish']].rename(columns={'finish':'Lf'})
bn=E[E.番手==1][['race_id','lk','finish']].rename(columns={'finish':'Bf'})
E=E.merge(ld[['lk','Lf']],on='lk',how='left').merge(bn[['lk','Bf']],on='lk',how='left')
fol=(E.番手==1)|(E['3番手+']==1)
E['ev_follow']=(fol&(E.Lf<=3)).astype(float); E['ev_chigirare']=(fol&(E.Lf<=3)&(E.finish>=E.Lf+3)).astype(float)
E['ev_lead']=((E.先頭==1)&(E.finish<=3)&E.Bf.notna()).astype(float); E['ev_chigiri']=((E.先頭==1)&(E.finish<=3)&(E.Bf>=E.finish+3)).astype(float)
E['ev_makuri']=(E.kimarite.astype(str).str.contains('捲')&(E.finish<=2)).astype(float); E['ev_race']=E.finish.notna().astype(float)
E['agari']=pd.to_numeric(E.agari,errors='coerce'); E['ev_agari']=E.agari-E.groupby('race_id').agari.transform('median')
E['ev_agari_n']=E.ev_agari.notna().astype(float); E['ev_agari']=E.ev_agari.fillna(0)
E['gotB']=E.sb.astype(str).str.contains('B').astype(float); E['ev_Bfin']=(E.gotB*E.finish.fillna(0)); 
ev=['ev_follow','ev_chigirare','ev_lead','ev_chigiri','ev_makuri','ev_race','ev_agari','ev_agari_n','gotB','ev_Bfin']
# 選手×日で合計 → 90日の窓（当日は含めない）
g=E.groupby(['rider_id','dt'])[ev].sum().reset_index().sort_values(['rider_id','dt'])
R=[]
for rid,x in g.groupby('rider_id'):
    x=x.set_index('dt')[ev]; r=x.rolling('90D',closed='left').sum(); r['rider_id']=rid; R.append(r.reset_index())
R=pd.concat(R)
E=E.merge(R,on=['rider_id','dt'],how='left',suffixes=('','_90'))
f=lambda a,b: E[a+'_90']/E[b+'_90'].replace(0,np.nan)
E['ちぎられ率']=f('ev_chigirare','ev_follow'); E['ちぎり率']=f('ev_chigiri','ev_lead'); E['まくり率']=f('ev_makuri','ev_race')
E['まくり回数']=E['ev_makuri_90']; E['上がり差']=f('ev_agari','ev_agari_n'); E['B時の平均着順']=f('ev_Bfin','gotB')
E['得点差']=E.score-E.groupby('race_id').score.transform('mean'); E['得点順位']=E.groupby('race_id').score.rank(ascending=False)
NEW=['ちぎられ率','ちぎり率','まくり率','まくり回数','上がり差','B時の平均着順','得点差','得点順位']
E=E[E.groupby('race_id').finish.transform(lambda s:(s==1).sum()==1)]
E=E[E.groupby('race_id').gotB.transform('sum')==1]
tr=E.date<'2026-09-01'; half=E.date<'2026-05-01'
A2=A+NEW
E['pB']=np.nan
for a,b in ((tr&half,tr&~half),(tr&~half,tr&half)):
    m=lgb.train(P,lgb.Dataset(E.loc[a,A2],E.loc[a,'gotB']),300); E.loc[b,'pB']=m.predict(E.loc[b,A2])
m1=lgb.train(P,lgb.Dataset(E.loc[tr,A2],E.loc[tr,'gotB']),300); E.loc[~tr,'pB']=m1.predict(E.loc[~tr,A2])
E=stage2_features(E)
B2=B+NEW
m2=lgb.train(P,lgb.Dataset(E.loc[tr,B2],(E.loc[tr,'finish']==1).astype(float)),400)
E['p_v2']=m2.predict(E[B2]); E['p_v2']=E.p_v2/E.groupby('race_id').p_v2.transform('sum')
te=E[~tr]
t=te.loc[te.groupby('race_id').pB.idxmax()]; print('段階1（主導権）の的中',round(t.gotB.mean()*100,1),'%')
t=te.loc[te.groupby('race_id').p_v2.idxmax()]; print('段階2 本命の1着率',round((t.finish==1).mean()*100,1),'%  3着以内',round((t.finish<=3).mean()*100,1),'%')
imp=pd.Series(m2.feature_importance('gain'),index=B2); print('段階2 効いた特徴',(imp/imp.sum()).sort_values(ascending=False).head(10).round(3).to_dict())
E[['race_id','date','car_no','finish','pB','p_v2']+NEW].to_pickle(f'{S}/tenkai2.pkl')
