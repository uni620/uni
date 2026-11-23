import sqlite3,pandas as pd,numpy as np
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
PK=pd.read_pickle(f'{S}/hx_picks.pkl')
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
rids=tuple(p['race_id'] for p in PK)
od=pd.read_sql("select race_id,bet_type,combo,odds from odds where is_final=1 and bet_type in ('2車単','3連単') and odds>0",c)
od=od[od.race_id.isin(set(rids))].drop_duplicates(['race_id','bet_type','combo']); OD={(r,b,k):o for r,b,k,o in od.itertuples(index=False)}
pay=pd.read_sql("select race_id,bet_type,combo,payout from payouts where bet_type in ('2車単','3連単')",c); PAY={(r,b,k):v for r,b,k,v in pay.itertuples(index=False)}
has3=set(od[od.bet_type=='3連単'].race_id); has2=set(od[od.bet_type=='2車単'].race_id)
rows=[]
for p in PK:
    rid=p['race_id']
    if rid not in has2 or rid not in has3: continue
    h=p['head']; p2=pd.Series(p['p2']); p3=pd.Series(p['p3']); p2/=p2.sum(); p3/=p3.sum()
    s2=p2.sort_values(ascending=False).index.tolist(); s3=(p2+p3).sort_values(ascending=False).index.tolist()
    for u in s2[:3]:
        k=f'{h}>{u}'; rows.append((rid,p['month'],p['pr'],'2車単',k,p['pr']*p2[u],OD.get((rid,'2車単',k),np.nan),PAY.get((rid,'2車単',k),0)))
        for v in [t for t in s3 if t!=u][:3]:
            k3=f'{h}>{u}>{v}'; pp=p['pr']*p2[u]*p3[v]/max(1-p3[u],1e-6)
            rows.append((rid,p['month'],p['pr'],'3連単',k3,pp,OD.get((rid,'3連単',k3),np.nan),PAY.get((rid,'3連単',k3),0)))
B=pd.DataFrame(rows,columns=['race_id','月','pr','bt','combo','P','odds','pay']); B=B[B.odds.notna()]
B['EV']=B.P*B.odds; B.to_pickle(f'{S}/hx_cut.pkl')
X=B[B.pr>=0.6]
print('自信60%以上',X.race_id.nunique(),'R（6/11〜11/20, 確定オッズあり）')
def summ(x,lab):
    n=x.race_id.nunique(); hit=x[x.pay>0].race_id.nunique()
    return dict(切り方=lab,点=len(x),レース=n,レース的中率=round(hit/max(n,1)*100,1),回収率=round(x.pay.sum()/max(len(x),1),1))
for bt,floors in (('2車単',[0,2,3,4,5]),('3連単',[0,5,8,10,15,20])):
    x=X[X.bt==bt]; out=[]
    for f in floors: out.append(summ(x[x.odds>=f],f'{f}倍以上だけ'))
    for t in (0.6,0.7,0.8,0.9,1.0): out.append(summ(x[x.EV>=t],f'見込みEV{t}以上だけ'))
    print('---',bt); print(pd.DataFrame(out).to_string(index=False))
# 見込み確率の当たり具合（較正）
x=X.copy(); x['帯']=pd.qcut(x.P,5); print(x.groupby(['bt','帯'],observed=True).apply(lambda g:pd.Series({'見込み%':g.P.mean()*100,'実際%':(g.pay>0).mean()*100,'点':len(g)})).round(1))
