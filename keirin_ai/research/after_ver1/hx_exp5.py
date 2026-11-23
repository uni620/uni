from hx_lib import *
import sqlite3
E=pd.read_pickle(f'{S}/hx2.pkl'); C=pickle.load(open(f'{S}/hx_cols.pkl','rb'))
PFC,CC=C['PFC'],C['CC']; FEAT=[f for f in C['FEAT'] if f not in PFC]; base=FEAT+CC
out,pr,m=evaluate(E,base,ret=True); print(out)
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
od=pd.read_sql("select race_id,combo,odds from odds where is_final=1 and bet_type='2車単' and odds>0",c).drop_duplicates(['race_id','combo'])
od['car_no']=od.combo.str.split('>').str[0].astype(int); od['iq']=1/od.odds
Q=od.groupby(['race_id','car_no']).iq.sum().reset_index(); Q['q']=Q.iq/Q.groupby('race_id').iq.transform('sum')
Y=E.merge(pr,on=['race_id','car_no']).merge(Q[['race_id','car_no','q']],on=['race_id','car_no'])
ai=Y.loc[Y.groupby('race_id').p.idxmax()].set_index('race_id'); mk=Y.loc[Y.groupby('race_id').q.idxmax()].set_index('race_id')
R=pd.DataFrame({'ai':ai.win,'mk':mk.win,'same':ai.car_no==mk.car_no,'mkq':mk.q,'cls':ai.race_class_code if 'race_class_code' in ai else 0,
  'n':ai.n,'種別':ai.種別,'ai_pos':np.select([ai.先頭==1,ai.番手==1,ai.単騎==1],['先頭','番手','単騎'],'3番手'),
  'mk_pos':np.select([mk.先頭==1,mk.番手==1,mk.単騎==1],['先頭','番手','単騎'],'3番手')})
R['cls']=Y.groupby('race_id').pf_race_class_code.first().reindex(R.index) if 'pf_race_class_code' in Y else 0
print('レース',len(R),'AI',round(R.ai.mean()*100,1),'人気',round(R.mk.mean()*100,1),'一致率',round(R.same.mean()*100,1))
d=R[~R.same]; print('不一致',len(d),'AIの本命が1着',round(d.ai.mean()*100,1),'人気1番が1着',round(d.mk.mean()*100,1))
R['人気の自信']=pd.cut(R.mkq,[0,.3,.4,.5,1])
for col in ('cls','n','種別','人気の自信','mk_pos'):
    g=R.groupby(col,observed=True).agg(R=('ai','size'),AI=('ai','mean'),人気=('mk','mean'),一致=('same','mean'))
    g[['AI','人気','一致']]=(g[['AI','人気','一致']]*100).round(1); print(g,'\n')
print('不一致の内訳（AI本命の位置 × 人気1番の位置）')
g=d.groupby(['ai_pos','mk_pos']).agg(R=('ai','size'),AI=('ai','mean'),人気=('mk','mean')); g[['AI','人気']]=(g[['AI','人気']]*100).round(1); print(g[g.R>=30])
Y.to_pickle(f'{S}/hx_Y.pkl')
