from hx_lib import *
E,C=load(); PFC,CM,CC=C['PFC'],C['CM'],C['CC']; FEAT=[f for f in C['FEAT'] if f not in PFC]
print('基準', evaluate(E,FEAT),flush=True)
print('+PF', evaluate(E,FEAT+PFC),flush=True)
print('+連係対戦', evaluate(E,FEAT+CC),flush=True)
print('全部', evaluate(E,FEAT+PFC+CM+CC),flush=True)
# 人気の参考: 2車単 確定オッズの1番人気（評価だけに使う）
import sqlite3
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
od=pd.read_sql("select race_id,combo,odds from odds where is_final=1 and bet_type='2車単' and odds>0",c).drop_duplicates(['race_id','combo'])
od['car_no']=od.combo.str.split('>').str[0].astype(int); od['iq']=1/od.odds
Q=od.groupby(['race_id','car_no']).iq.sum().reset_index()
y=E[(E.date>='2026-09-01')&(E.date<'2026-11-21')].merge(Q,on=['race_id','car_no'],how='inner')
t=y.loc[y.groupby('race_id').iq.idxmax()]
print('参考 1番人気', round(t.win.mean()*100,2), [round(t[(t.date>=a)&(t.date<b)].win.mean()*100,1) for a,b in MONTHS], len(t))
