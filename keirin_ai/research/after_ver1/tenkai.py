"""展開AI（強さの数字は使わない）。段階1: 誰がB（バック先頭＝主導権）を取るか。段階2: その展開で誰が1着か。
学習 1/1〜8/31、確認 9/1〜11/21。段階1の確率は学習期間の中では2つに分けて交互に当てはめ（自分の答えを見ない）。"""
import sqlite3, numpy as np, pandas as pd, lightgbm as lgb
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite',timeout=60)
E=pd.read_sql("""select e.race_id,e.car_no,e.line_id,e.line_pos,e.line_size,e.scratched,e.finish,e.km_nige,e.km_makuri,e.km_sashi,e.km_mark,
  e.sb_s,e.sb_h,e.sb_b,e.n1,e.n2,e.n3,e.nout,r.date,r.race_class,i.sb from entries e join races r using(race_id)
  left join result_info i using(race_id,car_no) where r.has_result=1 and r.race_class not like '%ガールズ%'""",c)
scr=set(E[E.scratched==1].race_id); E=E[~E.race_id.isin(scr)].copy()
E['finish']=pd.to_numeric(E.finish,errors='coerce')
E=E[E.groupby('race_id').finish.transform(lambda s:(s==1).sum()==1)]
E['gotB']=E.sb.astype(str).str.contains('B').astype(float)
E=E[E.groupby('race_id').gotB.transform('sum')==1]          # Bが1人だけ記録されたレース
for k in ['km_nige','km_makuri','km_sashi','km_mark','sb_s','sb_h','sb_b','n1','n2','n3','nout']: E[k]=pd.to_numeric(E[k],errors='coerce').fillna(0)
km=E[['km_nige','km_makuri','km_sashi','km_mark']].sum(axis=1).replace(0,np.nan); st=E[['n1','n2','n3','nout']].sum(axis=1).replace(0,np.nan)
for k,lab in (('km_nige','逃'),('km_makuri','捲'),('km_sashi','差'),('km_mark','マ')): E[lab+'率']=E[k]/km
E['S率']=E.sb_s/st; E['B率']=E.sb_b/st; E['H率']=E.sb_h/st; E['自力率']=E['逃率'].fillna(0)+E['捲率'].fillna(0)
sz=E.line_size.fillna(1); ps=E.line_pos.fillna(1)
E['単騎']=(sz<=1).astype(float); E['先頭']=((sz>=2)&(ps==1)).astype(float); E['番手']=((sz>=2)&(ps==2)).astype(float); E['3番手+']=((sz>=2)&(ps>=3)).astype(float)
E['ライン人数']=sz; E['lk']=E.race_id+'_'+E.line_id.fillna(-E.car_no).astype(int).astype(str)
E['n']=E.groupby('race_id').car_no.transform('size'); E['ライン数']=E.groupby('race_id').lk.transform('nunique')
ld=E[(E.先頭==1)|(E.単騎==1)]
E['先行候補数']=E.race_id.map(ld.assign(f=(ld.B率>0.2)|(ld.逃率>0.4)).groupby('race_id').f.sum())
E['自分の先頭のB率']=E.lk.map(E[E.先頭==1].set_index('lk').B率)
E['自分の先頭の逃率']=E.lk.map(E[E.先頭==1].set_index('lk').逃率)
mx=ld.groupby('race_id').B率.max(); E['レース最大B率']=E.race_id.map(mx)
E['B率の順位']=E.groupby('race_id').B率.rank(ascending=False)
A=['逃率','捲率','差率','マ率','S率','B率','H率','自力率','単騎','先頭','番手','3番手+','ライン人数','car_no','n','ライン数','先行候補数','B率の順位','レース最大B率']
P=dict(objective='binary',learning_rate=0.05,num_leaves=15,min_data_in_leaf=50,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,seed=1)
tr=E.date<'2026-09-01'
def norm(p,rid): return p/ p.groupby(rid).transform('sum')
# 段階1: 交互当てはめ（学習期間）＋全学習→確認期間
E['pB']=np.nan
half=E.date<'2026-05-01'
for a,b in ((tr&half,tr&~half),(tr&~half,tr&half)):
    m=lgb.train(P,lgb.Dataset(E.loc[a,A],E.loc[a,'gotB']),300); E.loc[b,'pB']=m.predict(E.loc[b,A])
m1=lgb.train(P,lgb.Dataset(E.loc[tr,A],E.loc[tr,'gotB']),300); E.loc[~tr,'pB']=m1.predict(E.loc[~tr,A])
E['pB']=norm(E.pB,E.race_id)
te=E[~tr]; top=te.loc[te.groupby('race_id').pB.idxmax()]
print('段階1 確認期間: B（主導権）を取る選手の的中率',round(top.gotB.mean()*100,1),'%（',te.race_id.nunique(),'R、当てずっぽうなら約',round((1/te.n).mean()*100,1),'%）')
imp=pd.Series(m1.feature_importance('gain'),index=A); print('段階1 効いた特徴',(imp/imp.sum()).sort_values(ascending=False).head(6).round(2).to_dict())
# 段階2
E['自分のラインのpB']=E.groupby('lk').pB.transform('sum')
E['自分の先頭のpB']=E.lk.map(E[E.先頭==1].set_index('lk').pB).fillna(0)
L=E.groupby(['race_id','lk']).pB.sum().rename('lp').reset_index()
L['m1']=L.groupby('race_id').lp.transform('max'); L['m2']=L.groupby('race_id').lp.transform(lambda s: s.nlargest(2).iloc[-1] if len(s)>1 else 0)
L['other']=np.where(L.lp>=L.m1,L.m2,L.m1)
E['他ラインの最大pB']=E.lk.map(L.set_index('lk').other)
E['競り合い']=E.groupby('race_id').pB.transform(lambda s: 1-(s.max()))
B=A+['pB','自分のラインのpB','自分の先頭のpB','他ラインの最大pB','競り合い']
E['win']=(E.finish==1).astype(float)
m2=lgb.train(P,lgb.Dataset(E.loc[tr,B],E.loc[tr,'win']),400)
E['p_tenkai']=norm(pd.Series(m2.predict(E[B]),index=E.index),E.race_id)
imp=pd.Series(m2.feature_importance('gain'),index=B); print('段階2 効いた特徴',(imp/imp.sum()).sort_values(ascending=False).head(8).round(2).to_dict())
E[['race_id','date','car_no','finish','pB','gotB','p_tenkai','line_id','line_pos','line_size']].to_pickle(f'{S}/tenkai.pkl')
te=E[~tr]; top=te.loc[te.groupby('race_id').p_tenkai.idxmax()]
print('段階2 確認期間: 本命の1着率',round((top.finish==1).mean()*100,1),'%')
