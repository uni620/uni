"""3連単20〜100倍（払戻2,000〜10,000円）の決着は、なぜ起きるのか。20倍未満・100倍超と比べる。"""
import sqlite3,pandas as pd,numpy as np
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
R=pd.read_pickle('ana_R.pkl').set_index('race_id'); E=pd.read_pickle('ana_E.pkl')
K=pd.read_sql("select e.race_id,e.car_no,e.kimarite,i.sb from entries e left join result_info i using(race_id,car_no)",c)
E=E.merge(K,on=['race_id','car_no'],how='left'); E=E[E.race_id.isin(R.index)].copy()
E['finish']=pd.to_numeric(E.finish,errors='coerce'); E['score']=pd.to_numeric(E.score,errors='coerce')
E['lk']=E.race_id+'_'+E.line_id.fillna(-E.car_no).astype(int).astype(str); E['sz']=E.line_size.fillna(1); E['pos']=E.line_pos.fillna(1)
E['得点順位']=E.groupby('race_id').score.rank(ascending=False,method='first')
E['位置']=np.select([E.sz<=1,E.pos==1,E.pos==2],['単騎','先頭','番手'],'3番手以降')
E['B']=E.sb.astype(str).str.contains('B')
# 人気（2車単の確定オッズから1着の確率）
od=pd.read_sql("select race_id,combo,odds from odds where is_final=1 and bet_type='2車単' and odds>0",c).drop_duplicates(['race_id','combo'])
od['car_no']=od.combo.str.split('>').str[0].astype(int); od['iq']=1/od.odds
Q=od.groupby(['race_id','car_no']).iq.sum().reset_index(); Q['人気']=Q.groupby('race_id').iq.rank(ascending=False,method='first')
E=E.merge(Q[['race_id','car_no','人気']],on=['race_id','car_no'],how='left')
R['帯']=pd.cut(R.p3,[0,2000,10000,1e9],labels=['20倍未満','20〜100倍','100倍超'],right=False)
f=lambda k:E[E.finish==k].drop_duplicates('race_id').set_index('race_id')
w,s,t=f(1),f(2),f(3)
R=R.join(w[['得点順位','位置','kimarite','B','lk','人気','car_no']].add_prefix('1着')).join(s[['得点順位','位置','lk','人気']].add_prefix('2着')).join(t[['得点順位','lk','人気']].add_prefix('3着'))
R=R.dropna(subset=['1着得点順位','2着得点順位','3着得点順位'])
# 本命の選手（得点1位）、人気1番がどうなったか
fav=E[E.得点順位==1].set_index('race_id'); R['得点1位の着順']=fav.finish.reindex(R.index).clip(upper=4).map({1:'1着',2:'2着',3:'3着',4:'4着以下'})
pf=E[E.人気==1].set_index('race_id'); R['1番人気の着順']=pf.finish.reindex(R.index).clip(upper=4).map({1:'1着',2:'2着',3:'3着',4:'4着以下'})
R['1-2着']=np.where(R['1着lk']==R['2着lk'],'同じライン','別ライン')
R['3着']=np.where(R['3着lk']==R['1着lk'],'1着と同じライン',np.where(R['3着lk']==R['2着lk'],'2着と同じライン','第3のライン'))
R['上位3人(得点)の3着内']=((R[['1着得点順位','2着得点順位','3着得点順位']]<=3).sum(axis=1)).astype(int).astype(str)+'人'
R['1着の得点順位']=R['1着得点順位'].clip(upper=5).astype(int).astype(str).replace({'5':'5位〜'})
R['決まり手']=R['1着kimarite'].fillna('').str[:1].map({'逃':'逃げ','捲':'捲り','差':'差し','マ':'マーク','ク':'マーク'}).fillna('他')
R['1着がB']=R['1着B'].map({True:'Bを取った',False:'取っていない'})
R['頭数']=R.groupby(level=0).size().reindex(R.index)  # dummy
n=E.groupby('race_id').size(); R['頭数']=n.reindex(R.index).map(lambda v:'9車' if v>=9 else ('7車' if v==7 else f'{v}車'))
R['クラス']=R.race_class.astype(str).map(lambda s:'S級' if 'S級' in s else ('A級チャレンジ' if 'チャレンジ' in s else ('A級' if 'A級' in s else s)))
print('全',len(R),'R  帯の割合', (R.帯.value_counts(normalize=True)*100).round(1).to_dict())
cols=['1着の得点順位','上位3人(得点)の3着内','得点1位の着順','1着位置','2着位置','1-2着','3着','決まり手','1着がB','並び','頭数','クラス','グレード']
for col in cols:
    t=pd.crosstab(R[col],R.帯,normalize='columns').mul(100).round(1); print(f'--- {col}（各帯の中の割合 %）'); print(t.to_string())
# 人気（6月以降）
Rq=R.dropna(subset=['1着人気','2着人気','3着人気']).copy()
for k in ('1着人気','2着人気','3着人気'): Rq[k+'帯']=Rq[k].clip(upper=5).astype(int).astype(str).replace({'5':'5〜'})
print('\n人気のわかるレース',len(Rq))
for col in ('1着人気帯','2着人気帯','3着人気帯','1番人気の着順'):
    print(f'--- {col}'); print(pd.crosstab(Rq[col] if col in Rq else R[col],Rq.帯 if col in Rq else R.帯,normalize='columns').mul(100).round(1).to_string())
Rq['型']=Rq['1着人気帯']+'-'+Rq['2着人気帯']+'-'+Rq['3着人気帯']
m=Rq[Rq.帯=='20〜100倍'].型.value_counts(normalize=True).mul(100).round(1).head(12)
print('--- 20〜100倍の決着で多い「人気の順番」の型（1着-2着-3着の人気）'); print(m.to_string())
R.to_pickle('why_mid_R.pkl')
