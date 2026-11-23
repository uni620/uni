import sqlite3,pandas as pd,numpy as np
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
AR=pd.read_pickle('ana_R.pkl').set_index('race_id')
od=pd.read_sql("select race_id,combo,odds from odds where is_final=1 and bet_type='3連単' and odds>0",c).drop_duplicates(['race_id','combo']); od=od[od.race_id.isin(AR.index)]
pay=pd.read_sql("select race_id,combo,payout from payouts where bet_type='3連単'",c); PAY=dict(zip(zip(pay.race_id,pay.combo),pay.payout))
ex=pd.read_sql("select race_id,combo,odds from odds where is_final=1 and bet_type='2車単' and odds>0",c).drop_duplicates(['race_id','combo'])
ex['car']=ex.combo.str.split('>').str[0].astype(int); ex['iq']=1/ex.odds
W=ex.groupby(['race_id','car']).iq.sum().reset_index(); W['pw']=W.iq/W.groupby('race_id').iq.transform('sum'); W['人気']=W.groupby('race_id').pw.rank(ascending=False,method='first')
fav=W.groupby('race_id').pw.max(); kon=set(fav[fav<0.35].index)
E=pd.read_pickle('hx2.pkl')[['race_id','car_no','lkey','line_pos']]
LK=dict(zip(zip(E.race_id,E.car_no),E.lkey)); LP=dict(zip(zip(E.race_id,E.car_no),E.line_pos)); POP=dict(zip(zip(W.race_id,W.car),W.人気))
B=od[od.race_id.isin(kon)&(od.odds>=20)&(od.odds<=100)].copy()
cs=B.combo.str.split('>',expand=True).astype(int); B['a'],B['b'],B['c']=cs[0],cs[1],cs[2]
r=B.race_id.values
B['pa']=[POP.get((x,y),9) for x,y in zip(r,B.a)]; B['pb']=[POP.get((x,y),9) for x,y in zip(r,B.b)]; B['pc']=[POP.get((x,y),9) for x,y in zip(r,B.c)]
la=[LK.get((x,y)) for x,y in zip(r,B.a)]; lb=[LK.get((x,y)) for x,y in zip(r,B.b)]; lc=[LK.get((x,y)) for x,y in zip(r,B.c)]
B['スジ12']=[u==v for u,v in zip(la,lb)]; B['3着第3ライン']=[(w!=u) and (w!=v) for u,v,w in zip(la,lb,lc)]
B['番手が2着']=[u==v and LP.get((x,bb),0)==LP.get((x,aa),0)+1 for x,aa,bb,u,v in zip(r,B.a,B.b,la,lb)]
B['pay']=[PAY.get(k,0) for k in zip(B.race_id,B.combo)]
days=AR.loc[list(kon)].date.nunique() if False else AR[AR.index.isin(kon)].date.nunique()
rules={'20〜100倍を全部':B.a>0,
 'ルール1: 1着1〜3番人気':B.pa<=3,
 'ルール2: 1着1〜3番人気 × 3着は第3のライン':(B.pa<=3)&B['3着第3ライン'],
 'ルール3: 1着1〜3番人気 × (2着が1着の番手 か 2着1〜4番人気)':(B.pa<=3)&(B.番手が2着|(B.pb<=4)),
 'ルール4: ルール3 × 3着は第3のラインか4番人気以下':(B.pa<=3)&(B.番手が2着|(B.pb<=4))&(B['3着第3ライン']|(B.pc>=4)),
 'ルール5: 1着1〜3番人気 × 2着が1着の番手':(B.pa<=3)&B.番手が2着,
 'ルール6: 1着1〜2番人気 × 3着4番人気以下':(B.pa<=2)&(B.pc>=4)}
rows=[]
for nm,m in rules.items():
    x=B[m]; g=x.groupby('race_id').pay.agg(['size','sum']); n=len(kon&set(B.race_id))
    rows.append((nm,round(g['size'].sum()/n,1),round((g['sum']>0).sum()/n*100,1),round(g['sum'].sum()/g['size'].sum(),1)))
print('人気で見た混戦（1番人気の勝率35%未満）',len(kon&set(B.race_id)),'R',round(len(kon&set(B.race_id))/days,1),'R/日（6〜11月）')
print(pd.DataFrame(rows,columns=['買い方','1Rの平均点数','中穴の的中率%','回収率%']).to_string(index=False))
