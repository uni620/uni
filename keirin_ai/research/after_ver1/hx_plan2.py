import sqlite3,pandas as pd,numpy as np,itertools
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
O=pd.read_pickle(f'{S}/hx_2nd.pkl'); H=pd.read_pickle(f'{S}/hx_conf.pkl')
E=pd.read_pickle(f'{S}/hx2.pkl')[['race_id','car_no','lkey','line_pos','先頭','番手','単騎']]
hh=H.loc[H.groupby('race_id').pr.idxmax()].set_index('race_id').join(E.set_index(['race_id','car_no']),on=['race_id','car_no'],rsuffix='_')
O=O.merge(E,on=['race_id','car_no']).sort_values(['race_id','s'],ascending=[True,False])
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
pay=pd.read_sql("select race_id,bet_type,combo,payout from payouts",c); pay=pay[pay.race_id.isin(hh.index)]
PAY={(r,b,k):v for r,b,k,v in pay.itertuples(index=False)}
G={r:g for r,g in O.groupby('race_id')}
def uniq(x):
    o=[]; [o.append(i) for i in x if i is not None and i not in o]; return o
rows=[]
for rid,h in hh.iterrows():
    if rid not in G: continue
    g=G[rid]; hd=int(h.car_no); ai=g.car_no.tolist()
    mate=g[(g.lkey==h.lkey)]
    if h.先頭==1: partner=mate[mate.line_pos==2].car_no.tolist()
    elif h.番手==1: partner=mate[mate.line_pos==1].car_no.tolist()
    else: partner=[]
    partner=partner[:1]
    rest_line=[c for c in mate.sort_values('line_pos').car_no if c not in partner]       # 同ラインの残り（3番手など）
    other=g[g.lkey!=h.lkey].car_no.tolist()                                              # 他ライン（AI順）
    sec_t=uniq(partner+other[:1]+ai)[:2]            # 展開: 相方 + 他ラインのAI1位
    thr_t=uniq(sec_t+rest_line[:1]+other[:2]+ai)[:4] # 3着候補: 2着候補 + 同ラインの残り + 他ラインAI上位
    sec_a=ai[:2]; thr_a=ai[:4]; sec3_a=ai[:3]
    sec3_t=uniq(partner+other[:2]+ai)[:3]
    P={'AI 2車単 2点':[('2車単',f'{hd}>{a}') for a in sec_a],
       '展開 2車単 2点':[('2車単',f'{hd}>{a}') for a in sec_t],
       'AI 2車単 3点':[('2車単',f'{hd}>{a}') for a in sec3_a],
       '展開 2車単 3点':[('2車単',f'{hd}>{a}') for a in sec3_t],
       'AI 3連単 6点':[('3連単',f'{hd}>{a}>{b}') for a in sec_a for b in thr_a if a!=b],
       '展開 3連単 6点':[('3連単',f'{hd}>{a}>{b}') for a in sec_t for b in thr_t if a!=b],
       'AI 3連単 9点':[('3連単',f'{hd}>{a}>{b}') for a in sec3_a for b in thr_a if a!=b],
       '展開 3連単 9点':[('3連単',f'{hd}>{a}>{b}') for a in sec3_t for b in uniq(sec3_t+rest_line[:1]+other+ai)[:4] if a!=b]}
    for nm,b in P.items():
        rows.append((rid,h.pr,nm,len(b)*100,sum(PAY.get((rid,t,k),0) for t,k in b),bool(partner)))
R=pd.DataFrame(rows,columns=['race_id','pr','買い方','cost','ret','相方あり'])
R['帯']=pd.cut(R.pr,[0,.5,.6,1],labels=['50%未満','50〜60%','60%以上']); R['hit']=R.ret>0
g=R.groupby(['帯','買い方'],observed=True).agg(R=('cost','size'),点=('cost','mean'),的中率=('hit','mean'),cost=('cost','sum'),ret=('ret','sum'))
g['点']=(g.点/100).round(1); g['的中率']=(g.的中率*100).round(1); g['回収率']=(g.ret/g.cost*100).round(1)
g['当たり中央値']=R[R.hit].groupby(['帯','買い方'],observed=True).ret.median()
pd.set_option('display.width',200); print(g[['R','点','的中率','回収率','当たり中央値']].to_string())
R.to_pickle(f'{S}/hx_plan2.pkl')
