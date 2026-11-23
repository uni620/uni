"""中穴の自信: AI（展開型）の3連単確率で、20〜100倍の目の上位k点が持つ確率の合計（=当たる見込み）。これが高いレースほど当たるか。9〜11月。"""
import sqlite3,pandas as pd,numpy as np
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
T=pd.read_pickle(f'{S}/tri_full.pkl')
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
od=pd.read_sql("select race_id,combo,odds from odds where is_final=1 and bet_type='3連単' and odds>0",c); od=od[od.race_id.isin(set(T.race_id))].drop_duplicates(['race_id','combo'])
pay=pd.read_sql("select race_id,combo,payout from payouts where bet_type='3連単'",c); PAY=dict(zip(zip(pay.race_id,pay.combo),pay.payout))
T=T.merge(od,on=['race_id','combo']); T['pay']=[PAY.get(k,0) for k in zip(T.race_id,T.combo)]
T['P']=T.P/T.groupby('race_id').P.transform('sum')                      # 頭の上位3人の範囲で合計1に
B=T[(T.odds>=20)&(T.odds<=100)].sort_values(['race_id','P'],ascending=[True,False]); B['rk']=B.groupby('race_id').cumcount()+1
AR=pd.read_pickle(f'{S}/ana_R.pkl').set_index('race_id'); D=pd.read_pickle(f'{S}/breakfeat.pkl')
rows=[]
for k in (5,10,15):
    x=B[B.rk<=k]; g=x.groupby('race_id').agg(見込み=('P','sum'),cost=('pay','size'),ret=('pay','sum'))
    g['hit']=g.ret>0; g['帯']=pd.qcut(g.見込み.rank(method='first'),5,labels=['低20%','2','3','4','高20%'])
    t=g.groupby('帯',observed=True).agg(R=('hit','size'),見込み=('見込み','mean'),的中率=('hit','mean'),c=('cost','sum'),r=('ret','sum'))
    t['回収率']=t.r/t.c/1; t=t[['R','見込み','的中率','回収率']]; t['見込み']=(t.見込み*100).round(1); t['的中率']=(t.的中率*100).round(1); t['回収率']=t.回収率.round(1)
    print(f'--- 20〜100倍の目をAI確率上位{k}点（中穴の自信で5段階）'); print(t.to_string())
    g.to_pickle(f'{S}/mid_conf_{k}.pkl')
