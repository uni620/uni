import sqlite3,pandas as pd,numpy as np,itertools
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
O=pd.read_pickle(f'{S}/hx_2nd.pkl'); H=pd.read_pickle(f'{S}/hx_conf.pkl')
hh=H.loc[H.groupby('race_id').pr.idxmax()].set_index('race_id')
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
pay=pd.read_sql("select race_id,bet_type,combo,payout from payouts",c); pay=pay[pay.race_id.isin(hh.index)]
PAY={(r,b,k):v for r,b,k,v in pay.itertuples(index=False)}
O=O.sort_values(['race_id','s'],ascending=[True,False])
opp=O.groupby('race_id').car_no.apply(list)
def plans(h,o):
    s=lambda a,b:f"{min(a,b)}-{max(a,b)}"; t=lambda *x:'-'.join(map(str,sorted(x)))
    P={'2車単 1点':[('2車単',f'{h}>{o[0]}')],
       '2車単 2点':[('2車単',f'{h}>{x}') for x in o[:2]],
       '2車単 3点':[('2車単',f'{h}>{x}') for x in o[:3]],
       '2車複 1点':[('2車複',s(h,o[0]))],
       '3連複 頭-相手3人 3点':[('3連複',t(h,a,b)) for a,b in itertools.combinations(o[:3],2)],
       '3連単 頭→2人→3人 4点':[('3連単',f'{h}>{a}>{b}') for a in o[:2] for b in o[:3] if a!=b],
       '3連単 頭→2人→4人 6点':[('3連単',f'{h}>{a}>{b}') for a in o[:2] for b in o[:4] if a!=b],
       '3連単 頭→3人→4人 9点':[('3連単',f'{h}>{a}>{b}') for a in o[:3] for b in o[:4] if a!=b],
       '3連単 1点':[('3連単',f'{h}>{o[0]}>{o[1]}')]}
    return P
rows=[]
for rid,x in hh.iterrows():
    if rid not in opp.index: continue
    for nm,bets in plans(int(x.car_no),opp[rid]).items():
        ret=sum(PAY.get((rid,b,k),0) for b,k in bets)
        rows.append((rid,x.date,x.pr,nm,len(bets)*100,ret))
R=pd.DataFrame(rows,columns=['race_id','date','pr','買い方','cost','ret'])
R['帯']=pd.cut(R.pr,[0,.5,.6,1],labels=['50%未満','50〜60%','60%以上'])
R['hit']=R.ret>0
g=R.groupby(['帯','買い方'],observed=True).agg(R=('cost','size'),的中率=('hit','mean'),cost=('cost','sum'),ret=('ret','sum'))
g['的中率']=(g.的中率*100).round(1); g['回収率']=(g.ret/g.cost*100).round(1)
hp=R[R.hit].groupby(['帯','買い方'],observed=True).ret.median(); g['当たり配当の中央値']=hp.round(0)
pd.set_option('display.width',200); print(g[['R','的中率','回収率','当たり配当の中央値']].to_string())
R.to_pickle(f'{S}/hx_plan.pkl')
