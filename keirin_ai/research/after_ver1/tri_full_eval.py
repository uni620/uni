import sqlite3,pandas as pd,numpy as np
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
T=pd.read_pickle(f'{S}/tri_full.pkl'); Xh=pd.read_pickle(f'{S}/mid_band.pkl')
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
rids=set(T.race_id)&set(Xh.race_id)
od=pd.read_sql("select race_id,combo,odds from odds where is_final=1 and bet_type='3連単' and odds>0",c); od=od[od.race_id.isin(rids)].drop_duplicates(['race_id','combo'])
pay=pd.read_sql("select race_id,combo,payout from payouts where bet_type='3連単'",c); PAY=dict(zip(zip(pay.race_id,pay.combo),pay.payout))
T=T[T.race_id.isin(rids)].merge(od,on=['race_id','combo'])
T['pay']=[PAY.get((r,k),0) for r,k in zip(T.race_id,T.combo)]; T['月']=T.race_id.map(dict(zip(Xh.race_id,Xh.月)))
B=T[(T.odds>=20)&(T.odds<=100)].sort_values(['race_id','P'],ascending=[True,False]); B['rank']=B.groupby('race_id').cumcount()+1
H=Xh[Xh.race_id.isin(rids)]
rng=np.random.default_rng(0)
def summ(x):
    g=x.groupby('race_id').agg(c=('pay','size'),r=('pay','sum'))
    s=[g.iloc[rng.integers(0,len(g),len(g))].pipe(lambda d:d.r.sum()/d.c.sum()) for _ in range(1000)]
    mo=x.groupby('月').apply(lambda d:int(round(d.pay.sum()/len(d)))).tolist()
    return f"的中{(g.r>0).mean()*100:.1f}% 回収{g.r.sum()/g.c.sum():.1f}% [{np.percentile(s,2.5):.0f}-{np.percentile(s,97.5):.0f}] 月{mo}"
print('共通レース',len(rids))
for k in (1,3,5,10,20):
    print(f'{k}点  展開型: {summ(B[B["rank"]<=k])}\n     簡単計算: {summ(H[H["rank"]<=k])}')
# 較正: 20〜100倍の目でのAI確率 vs 実際
B['帯']=pd.qcut(B.P,5); print((B.groupby('帯',observed=True).agg(見込み=('P','mean'),実際=('pay',lambda v:(v>0).mean()),点=('pay','size'))*[100,100,1]).round(2).to_string())
# 頭の上位3人で、実際の3連単が表にある率
fin=pd.read_sql("select race_id,car_no,finish from entries where finish<=3",c); fin=fin[fin.race_id.isin(rids)]
ans={r:'>'.join(map(str,g.sort_values('finish').car_no.astype(int))) for r,g in fin.groupby('race_id') if len(g)==3}
Tall=pd.read_pickle(f'{S}/tri_full.pkl'); Tall=Tall[Tall.race_id.isin(rids)]; Tall['rk']=Tall.groupby('race_id').P.rank(ascending=False)
hit=Tall[[ans.get(r)==k for r,k in zip(Tall.race_id,Tall.combo)]]
print('実際の3連単が AI確率の上位k位以内に入った率', {k:round((hit.rk<=k).sum()/len(ans)*100,1) for k in (1,5,10,20,50)})

# ---- 並び別（細切れ戦だけ など）
AR=pd.read_pickle(f'{S}/ana_R.pkl').set_index('race_id')
B['並び']=B.race_id.map(AR.並び); H=H.assign(並び=H.race_id.map(AR.並び))
print('\n=== 並び別（20〜100倍の目, AI確率上位k点）')
for nb in ('細切れ(4分以上)','3分戦','2分戦'):
    n=B[B.並び==nb].race_id.nunique(); r20=AR.loc[[r for r in rids if AR.loc[r,'並び']==nb] if False else [], :] if False else None
    print(f'--- {nb} {n}R')
    for k in (3,5,10,20):
        print(f'  {k}点 展開型: {summ(B[(B.並び==nb)&(B["rank"]<=k)])} | 簡単計算: {summ(H[(H.並び==nb)&(H["rank"]<=k)])}')
ans_odds=Xh.drop_duplicates('race_id').set_index('race_id').ans_odds
for nb in ('細切れ(4分以上)','3分戦','2分戦'):
    r=[x for x in rids if AR.loc[x,'並び']==nb and x in ans_odds.index]; a=ans_odds.loc[r]
    print(nb,'実際の結果が20〜100倍だった割合',round(((a>=20)&(a<=100)).mean()*100,1),'%')
