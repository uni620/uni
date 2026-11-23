import sqlite3,pandas as pd,numpy as np
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
Z=pd.read_pickle(f'{S}/tri_full_Z.pkl')
Z=Z[Z.hpr==Z.groupby('race_id').hpr.transform('max')]           # 頭 = 頭AIの1位
E=pd.read_pickle(f'{S}/hx2.pkl')[['race_id','car_no','lkey','line_pos']]
AR=pd.read_pickle(f'{S}/ana_R.pkl').set_index('race_id')
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
od=pd.read_sql("select race_id,combo,odds from odds where is_final=1 and bet_type='3連単' and odds>0",c); od=od[od.race_id.isin(set(Z.race_id))].drop_duplicates(['race_id','combo'])
OD={(r,k):o for r,k,o in od.itertuples(index=False)}
pay=pd.read_sql("select race_id,combo,payout from payouts where bet_type='3連単'",c); PAY=dict(zip(zip(pay.race_id,pay.combo),pay.payout))
LK=dict(zip(zip(E.race_id,E.car_no),E.lkey))
cls=pd.read_sql("select race_id,race_class from races",c).set_index('race_id').race_class
rows=[]
for rid,g in Z.groupby('race_id'):
    if not any((rid,k) in OD for k in [None]) and rid not in AR.index: continue
    h=int(g['head'].iloc[0]); hl=LK.get((rid,h))
    q2=(g.set_index('car_no').p2/g.p2.sum()); q3=(g.set_index('car_no').p3/g.p3.sum()); q2.index=q2.index.astype(int); q3.index=q3.index.astype(int)
    same=[x for x in q2.sort_values(ascending=False).index if LK.get((rid,x))==hl]
    diff=[x for x in q2.sort_values(ascending=False).index if LK.get((rid,x))!=hl]
    allo=list(q2.sort_values(ascending=False).index)
    third=lambda excl,n:[x for x in (q2+q3).sort_values(ascending=False).index if x not in excl][:n]
    P={}
    P['今の形: 2着AI上位3人→3着AI上位3人(9点)']=[(u,v) for u in allo[:3] for v in third([u],3)]
    for n2,n3 in ((2,3),(2,4),(3,4),(2,'全')):
        sec=diff[:n2]
        P[f'スジ違い: 2着別ライン{n2}人→3着{"AI上位"+str(n3)+"人" if n3!="全" else "全員"}']=[(u,v) for u in sec for v in (third([u],n3) if n3!='全' else [x for x in allo if x!=u])]
    P['スジ違い: 2着別ライン2人→3着は3つ目のライン+人気薄(AI下位)含む4人']=[(u,v) for u in diff[:2] for v in ([x for x in allo if x!=u and LK.get((rid,x)) not in (hl,LK.get((rid,u)))][:2]+[x for x in allo if x!=u][-2:])]
    for nm,bets in P.items():
        bets=list(dict.fromkeys((u,v) for u,v in bets if u!=v))
        ks=[f'{h}>{u}>{v}' for u,v in bets]
        hit=[k for k in ks if (rid,k) in PAY]
        rows.append((rid,nm,len(ks)*100,sum(PAY.get((rid,k),0) for k in ks),OD.get((rid,hit[0])) if hit else None))
X=pd.DataFrame(rows,columns=['race_id','買い方','cost','ret','hit_odds'])
X['並び']=X.race_id.map(AR.並び); X['月']=X.race_id.str[:6]; X['級']=X.race_id.map(cls).astype(str).map(lambda s:'S級' if 'S級' in s else ('A級チャレンジ' if 'チャレンジ' in s else 'A級'))
X['帯的中']=(X.hit_odds>=20)&(X.hit_odds<=100)
rng=np.random.default_rng(0)
def summ(x):
    s=[x.iloc[rng.integers(0,len(x),len(x))].pipe(lambda d:d.ret.sum()/d.cost.sum()*100) for _ in range(500)]
    return pd.Series({'R':len(x),'点':round(x.cost.mean()/100,1),'的中率':round((x.ret>0).mean()*100,1),'うち20-100倍':round(x.帯的中.mean()*100,1),
                      '回収率':round(x.ret.sum()/x.cost.sum()*100,1),'95%幅':f'{np.percentile(s,2.5):.0f}-{np.percentile(s,97.5):.0f}',
                      '月別':[int(round(d.ret.sum()/d.cost.sum()*100)) for _,d in x.groupby('月')]})
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',60)
print('=== 全レース'); print(X.groupby('買い方').apply(summ).to_string())
for col,vals in (('並び',['3分戦','細切れ(4分以上)']),('級',['S級','A級','A級チャレンジ'])):
    for v in vals:
        print(f'=== {col}={v}'); print(X[X[col]==v].groupby('買い方').apply(summ)[['R','点','的中率','うち20-100倍','回収率','95%幅']].to_string())
X.to_pickle(f'{S}/suji_chigai.pkl')
