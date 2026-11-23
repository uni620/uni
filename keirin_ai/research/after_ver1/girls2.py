"""ガールズ専用の頭AI＋相手（オッズなし）。各月その月より前のガールズ（とその選手の過去）で学習。4〜11月で評価。"""
import sqlite3,pandas as pd,numpy as np,lightgbm as lgb,warnings,itertools
warnings.filterwarnings('ignore')
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
E=pd.read_sql("""select e.race_id,e.car_no,e.rider_id,e.score,e.finish,e.scratched,e.win_rate,e.ren2,e.ren3,e.km_nige,e.km_makuri,e.km_sashi,e.km_mark,
 e.sb_s,e.sb_h,e.sb_b,e.n1,e.n2,e.n3,e.nout,e.gear,e.age,r.date,r.venue,r.race_class,r.race_name,r.bank_len,i.agari,i.sb
 from entries e join races r using(race_id) left join result_info i using(race_id,car_no) where r.race_class like '%ガールズ%' and r.has_result=1""",c)
for k in ['score','finish','win_rate','ren2','ren3','km_nige','km_makuri','km_sashi','km_mark','sb_s','sb_h','sb_b','n1','n2','n3','nout','gear','age','agari','bank_len']: E[k]=pd.to_numeric(E[k],errors='coerce')
E=E[~E.race_id.isin(E[E.scratched==1].race_id)]
E=E[E.groupby('race_id').finish.transform(lambda s:(s==1).sum()==1)].copy()
E['dt']=pd.to_datetime(E.date); E['n']=E.groupby('race_id').car_no.transform('size')
st=E[['n1','n2','n3','nout']].sum(axis=1).replace(0,np.nan); km=E[['km_nige','km_makuri','km_sashi','km_mark']].sum(axis=1).replace(0,np.nan)
E['B率']=E.sb_b/st; E['S率']=E.sb_s/st; E['逃率']=E.km_nige/km; E['捲率']=E.km_makuri/km; E['差率']=E.km_sashi/km
for k in ['score','win_rate','ren2','ren3','B率','S率','逃率','捲率']:
    E[k+'_順位']=E.groupby('race_id')[k].rank(ascending=False); E[k+'_差']=E[k]-E.groupby('race_id')[k].transform('max')
E['得点2位との差']=E.score-E.groupby('race_id').score.transform(lambda s:s.nlargest(2).iloc[-1])
# 近況（過去の自分の着順・上がり、全レース種別のうちガールズだけ）
E=E.sort_values(['rider_id','dt'])
g=E.groupby('rider_id')
E['fin_pct']=E.finish/E.n; E['win']=(E.finish==1).astype(float)
E['agari_rk']=E.groupby('race_id').agari.rank(pct=True)
for col,new in (('fin_pct','近5着'),('win','近10勝率'),('agari_rk','近5上がり')):
    s=g[col].shift(1); w=10 if new=='近10勝率' else 5
    E[new]=s.groupby(E.rider_id).transform(lambda v:v.rolling(w,min_periods=2).mean())
for k in ['近5着','近10勝率','近5上がり']: E[k+'_順位']=E.groupby('race_id')[k].rank(ascending=(k!='近10勝率'))
F=[c for c in E.columns if c.endswith(('_順位','_差'))]+['score','win_rate','ren2','ren3','B率','S率','逃率','捲率','差率','gear','age','n','car_no','bank_len','得点2位との差','近5着','近10勝率','近5上がり']
P=dict(objective='lambdarank',learning_rate=0.03,num_leaves=15,min_data_in_leaf=50,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,eval_at=[1],lambdarank_truncation_level=3)
PB=dict(objective='binary',learning_rate=0.03,num_leaves=15,min_data_in_leaf=50,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1)
pay=pd.read_sql("select race_id,bet_type,combo,payout from payouts where bet_type in ('2車単','3連単','2車複','3連複','ワイド')",c); PAY={(r,b,k):v for r,b,k,v in pay.itertuples(index=False)}
E=E.sort_values(['race_id','car_no']).reset_index(drop=True)
rows=[]
for m in [f'2026-{i:02d}' for i in range(4,12)]:
    tr=E[E.date<m+'-01']; te=E[E.date.str[:7]==m].copy()
    if te.empty: continue
    lab=np.select([tr.finish==1,tr.finish==2,tr.finish==3],[3,2,1],0)
    d=lgb.Dataset(tr[F],lab); d.set_group(tr.groupby('race_id',sort=False).size().values)
    te['s']=sum(lgb.train(dict(P,seed=s),d,400).predict(te[F]) for s in (1,2))
    te['pr']=sum(lgb.train(dict(PB,seed=s),lgb.Dataset(tr[F],tr.win),400).predict(te[F]) for s in (1,2)); te['pr']/=te.groupby('race_id').pr.transform('sum')
    for rid,gg in te.groupby('race_id'):
        gg=gg.sort_values('s',ascending=False); o=gg.car_no.astype(int).tolist(); h=o[0]; prh=gg.pr.max()
        sc=gg.sort_values('score',ascending=False).car_no.astype(int).tolist()
        bets={'2車単1点':[('2車単',f'{h}>{o[1]}')],'2車単2点':[('2車単',f'{h}>{x}') for x in o[1:3]],'2車単3点':[('2車単',f'{h}>{x}') for x in o[1:4]],
              '2車複1点':[('2車複','-'.join(map(str,sorted((h,o[1])))))],'3連複1点':[('3連複','-'.join(map(str,sorted(o[:3]))))],
              '3連単4点':[('3連単',f'{h}>{a}>{b}') for a in o[1:3] for b in o[1:4] if a!=b],'3連単6点':[('3連単',f'{h}>{a}>{b}') for a in o[1:4] for b in o[1:4] if a!=b],
              'ワイド1点':[('ワイド','-'.join(map(str,sorted((h,o[1])))))],'ワイド2点':[('ワイド','-'.join(map(str,sorted((h,x))))) for x in o[1:3]],'ワイド3点':[('ワイド','-'.join(map(str,sorted((h,x))))) for x in o[1:4]],
              '3連複3点':[('3連複','-'.join(map(str,sorted((h,a,b))))) for a,b in itertools.combinations(o[1:4],2)]}
        for nm,bs in bets.items():
            rows.append((m,rid,prh,int(gg.finish.iloc[0]==1),int(sc[0]==h),nm,len(bs)*100,sum(PAY.get((rid,b,k),0) for b,k in bs)))
    print(m,'done',flush=True)
R=pd.DataFrame(rows,columns=['月','race_id','pr','headwin','頭=得点1位','買い方','cost','ret']); R.to_pickle('girls_R2.pkl')
h=R.drop_duplicates('race_id'); print('ガールズ',len(h),'R  AI頭の1着率',round(h.headwin.mean()*100,1),'%  頭が得点1位と同じ',round(h['頭=得点1位'].mean()*100,1),'%')
R['帯']=pd.cut(R.pr,[0,.6,.7,.8,.9,1])
print(R.drop_duplicates('race_id').groupby('帯',observed=True).agg(R=('headwin','size'),頭=('headwin','mean')).round(3).to_string())
t=R.groupby(['帯','買い方'],observed=True).apply(lambda d:pd.Series({'的中':round((d.ret>0).mean()*100,1),'回収':round(d.ret.sum()/d.cost.sum()*100,1)}))
print(t.unstack('買い方').to_string())
