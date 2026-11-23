import sqlite3,pandas as pd,numpy as np,itertools
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
R=pd.read_pickle('ana_R.pkl'); E=pd.read_pickle('ana_E.pkl'); E=E[E.race_id.isin(R.race_id)].copy()
P=pd.read_sql("select race_id,bet_type,combo,payout from payouts where bet_type in ('3連単','2車単')",c)
PAY={(r,b,k):v for r,b,k,v in P.itertuples(index=False)}
E['score']=pd.to_numeric(E.score,errors='coerce'); E['lk']=E.race_id+'_'+E.line_id.fillna(-E.car_no).astype(int).astype(str)
E['sz']=E.line_size.fillna(1); E['pos']=E.line_pos.fillna(1)
G={r:g for r,g in E.groupby('race_id')}; RI=R.set_index('race_id')
rng=np.random.default_rng(0)
res=[]
def add(name,rid,bets):
    if not bets: return
    res.append((name,rid,RI.at[rid,'並び'],RI.at[rid,'date'][:7],len(bets)*100,sum(PAY.get((rid,'3連単',k),0) for k in bets)))
for rid,g in G.items():
    g=g.sort_values('car_no'); cars=g.car_no.tolist(); sc=g.set_index('car_no').score
    top2=sc.sort_values(ascending=False).index[:2].tolist(); lk=g.set_index('car_no').lk
    # スジ違い: 得点上位2人が別ライン → 各ラインの得点最上位を1-2着候補、残り全員3着
    if len(top2)==2 and lk[top2[0]]!=lk[top2[1]]:
        heads=g.loc[g.groupby('lk').score.idxmax()]; heads=heads[heads.sz>=2].car_no.tolist()
        if len(heads)>=2:
            rest=[x for x in cars if x not in heads]
            add('スジ違い（各ライン得点1位−同−その他）',rid,[f'{a}>{b}>{x}' for a in heads for b in heads if a!=b for x in rest])
    # サンドイッチ: 3車ラインがちょうど1つ、その先頭か番手の得点が低い（2人の差3点以上）→ 先頭・番手−他ライン全員−先頭・番手
    L3=g[g.sz>=3].lk.unique()
    if len(L3)==1:
        m=g[g.lk==L3[0]].sort_values('pos'); a,b=m.car_no.iloc[0],m.car_no.iloc[1]
        if abs(sc[a]-sc[b])>=3:
            oth=[x for x in cars if lk[x]!=L3[0]]
            add('サンドイッチ（3車の1-2番手−他ライン−1-2番手）',rid,[f'{x}>{o}>{y}' for x,y in ((a,b),(b,a)) for o in oth])
    # ヨーロッパ: 得点上位の先頭の番手が4/6/8番 → その番手1着固定流し
    for _,ld in g[(g.pos==1)&(g.sz>=2)].iterrows():
        if ld.car_no in top2:
            bn=g[(g.lk==ld.lk)&(g.pos==2)]
            if len(bn) and int(bn.car_no.iloc[0]) in (4,6,8):
                h=int(bn.car_no.iloc[0]); o=[x for x in cars if x!=h]
                add('ヨーロッパ（得点上位先頭の番手4/6/8を1着流し）',rid,[f'{h}>{u}>{v}' for u in o for v in o if u!=v])
    # B回数最多（得点上位2人以外）を1着流し
    nt=g[~g.car_no.isin(top2)]
    if len(nt):
        bmax=nt.sort_values('sb_b',ascending=False).iloc[0]
        if bmax.sb_b>=g.sb_b.max():
            h=int(bmax.car_no); o=[x for x in cars if x!=h]
            add('B回数最多（得点上位2人以外）を1着流し',rid,[f'{h}>{u}>{v}' for u in o for v in o if u!=v])
    # 細切れ戦の単騎で戦法が他と違う（唯一の差し/捲り型）を1着流し
    if RI.at[rid,'並び'].startswith('細切れ'):
        ld=g[g.pos==1].copy(); km=ld[['km_nige','km_makuri','km_sashi','km_mark']].apply(pd.to_numeric,errors='coerce').fillna(0)
        ld['型']=km.idxmax(axis=1)
        for _,s in ld[ld.sz<=1].iterrows():
            if (ld.型==s.型).sum()==1:
                h=int(s.car_no); o=[x for x in cars if x!=h]
                add('細切れ戦で戦法が唯一の単騎を1着流し',rid,[f'{h}>{u}>{v}' for u in o for v in o if u!=v])
X=pd.DataFrame(res,columns=['買い方','race_id','並び','月','cost','ret'])
def summ(x):
    r=x.groupby('race_id')[['cost','ret']].sum(); s=[r.iloc[rng.integers(0,len(r),len(r))].pipe(lambda d:d.ret.sum()/d.cost.sum()*100) for _ in range(1000)]
    return pd.Series({'R数':len(r),'1R点':round(x.cost.sum()/len(r)/100,1),'的中率':round((r.ret>0).mean()*100,1),'回収率':round(r.ret.sum()/r.cost.sum()*100,1),'95%幅':f'{np.percentile(s,2.5):.0f}-{np.percentile(s,97.5):.0f}'})
pd.set_option('display.width',200)
print(X.groupby('買い方').apply(summ).to_string()); print()
print(X.groupby(['買い方','並び']).apply(summ).to_string()); print()
m=X.groupby(['買い方','月']).apply(lambda d:round(d.ret.sum()/d.cost.sum()*100)).unstack(); print(m.to_string())
