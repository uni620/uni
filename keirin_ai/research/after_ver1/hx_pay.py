import sqlite3,pandas as pd,numpy as np
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
Y=pd.read_pickle(f'{S}/hx_conf.pkl')
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
pay=pd.read_sql("select race_id,bet_type,combo,payout from payouts",c); pay=pay[pay.race_id.isin(Y.race_id)]
fin=pd.read_sql("select race_id,car_no,finish from entries",c); fin=fin[fin.race_id.isin(Y.race_id)]
t=Y.loc[Y.groupby('race_id').pr.idxmax()].copy()
t['帯']=pd.cut(t.pr,[0,.5,.6,1],labels=['50%未満','50〜60%','60%以上'])
n_r=Y.groupby('race_id').size()
PAY={(r,b,k):v for r,b,k,v in pay.itertuples(index=False)}
P2=pay[pay.bet_type=='2車単'].drop_duplicates('race_id').set_index('race_id').payout
P3=pay[pay.bet_type=='3連単'].drop_duplicates('race_id').set_index('race_id').payout
rows=[]
for x in t.itertuples():
    n=n_r[x.race_id]; h=x.car_no
    c2=(n-1)*100; c3=(n-1)*(n-2)*100
    r2=P2.get(x.race_id,0) if x.win else 0; r3=P3.get(x.race_id,0) if x.win else 0
    rows.append((x.帯,x.win,P2.get(x.race_id,np.nan),P3.get(x.race_id,np.nan),c2,r2,c3,r3))
R=pd.DataFrame(rows,columns=['帯','win','p2','p3','c2','r2','c3','r3'])
for b,g in R.groupby('帯',observed=True):
    h=g[g.win==1]
    print(f"[{b}] {len(g)}R 頭的中 {g.win.mean()*100:.1f}%")
    print(f"  当たった時の2車単 中央値 {h.p2.median():.0f}円 平均 {h.p2.mean():.0f}円 (25%〜75%: {h.p2.quantile(.25):.0f}〜{h.p2.quantile(.75):.0f})")
    print(f"  当たった時の3連単 中央値 {h.p3.median():.0f}円 平均 {h.p3.mean():.0f}円 (25%〜75%: {h.p3.quantile(.25):.0f}〜{h.p3.quantile(.75):.0f})")
    print(f"  2車単 頭流し({g.c2.mean()/100:.0f}点) 回収率 {g.r2.sum()/g.c2.sum()*100:.1f}% | 3連単 頭流し({g.c3.mean()/100:.0f}点) 回収率 {g.r3.sum()/g.c3.sum()*100:.1f}%")
