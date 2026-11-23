"""3連単 20〜100倍の目を当てる。AIの確率（頭AIの1着確率から Harville で3連単の確率、オッズ不使用）→ オッズ20〜100倍の目だけ → AI確率の高い順に k 点。9〜11月（学習外）。"""
import sqlite3,itertools,pandas as pd,numpy as np
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
H=pd.read_pickle(f'{S}/hx_conf.pkl')     # race_id,date,car_no,win,pr（各月その月より前で学習）
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
od=pd.read_sql("select race_id,combo,odds from odds where is_final=1 and bet_type='3連単' and odds>0",c)
od=od[od.race_id.isin(set(H.race_id))].drop_duplicates(['race_id','combo'])
pay=pd.read_sql("select race_id,combo,payout from payouts where bet_type='3連単'",c); PAY=dict(zip(zip(pay.race_id,pay.combo),pay.payout))
fin=pd.read_sql("select race_id,car_no,finish from entries where finish<=3",c); fin=fin[fin.race_id.isin(set(H.race_id))]
ANS={r:'>'.join(map(str,g.sort_values('finish').car_no.astype(int))) for r,g in fin.groupby('race_id') if len(g)==3}
ODS={r:dict(zip(g.combo,g.odds)) for r,g in od.groupby('race_id')}
rows=[]
for rid,g in H.groupby('race_id'):
    if rid not in ODS or rid not in ANS: continue
    p=dict(zip(g.car_no.astype(int),g.pr)); cars=list(p); O=ODS[rid]; ans=ANS[rid]
    P={}
    for a,b,d in itertools.permutations(cars,3):
        P[f'{a}>{b}>{d}']=p[a]*p[b]/(1-p[a])*p[d]/(1-p[a]-p[b])
    band=[(k,P[k],O[k]) for k in P if k in O and 20<=O[k]<=100]
    band.sort(key=lambda t:-t[1])
    ao=O.get(ans,np.nan); top=max(g.pr)
    for i,(k,pp,o) in enumerate(band[:20]):
        rows.append((rid,g.date.iloc[0][:7],top,i+1,k,pp,o,PAY.get((rid,k),0) if k==ans else 0,ao))
X=pd.DataFrame(rows,columns=['race_id','月','top','rank','combo','P','odds','pay','ans_odds'])
X.to_pickle(f'{S}/mid_band.pkl')
R=X.drop_duplicates('race_id'); print('レース',len(R),' 実際の結果が20〜100倍だった割合',round(((R.ans_odds>=20)&(R.ans_odds<=100)).mean()*100,1),'%')
rng=np.random.default_rng(0)
def summ(x):
    g=x.groupby('race_id').agg(c=('pay','size'),r=('pay','sum')); s=[g.iloc[rng.integers(0,len(g),len(g))].pipe(lambda d:d.r.sum()/d.c.sum()*100) for _ in range(1000)]
    mo=x.groupby('月').apply(lambda d:int(round(d.pay.sum()/len(d)))).tolist()
    return pd.Series({'R':len(g),'点/R':round(len(x)/len(g),1),'的中率%':round((g.r>0).mean()*100,1),'回収率%':round(g.r.sum()/g.c.sum()*100,1),'95%幅':f'{np.percentile(s,2.5):.0f}-{np.percentile(s,97.5):.0f}','月別9-11':mo})
out={}
for k in (1,3,5,10,20): out[f'AI確率上位{k}点（全レース）']=summ(X[X['rank']<=k])
for k in (3,5,10):
    for lo,hi,lab in ((0,.4,'頭の自信40%未満'),(.4,.6,'頭の自信40〜60%'),(.6,1,'頭の自信60%以上')):
        x=X[(X['rank']<=k)&(X.top>=lo)&(X.top<hi)]; out[f'上位{k}点・{lab}']=summ(x)
print(pd.DataFrame(out).T.to_string())
# 比較: 20〜100倍の目を無作為にk点
for k in (3,10):
    x=X.groupby('race_id').apply(lambda g:g.sample(min(k,len(g)),random_state=0)).reset_index(drop=True); print(f'比較: 20〜100倍から無作為{k}点',summ(x).to_dict())
