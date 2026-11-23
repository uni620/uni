import sys, sqlite3, pandas as pd, numpy as np
sys.path.insert(0,'/home/user/uni/keirin_ai')
from keirin.combos import combo_probs
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
t=pd.read_pickle(f'{S}/wf_v2.pkl'); c=sqlite3.connect(f'{S}/wf.sqlite')
P={(r.race_id,r.combo):r.payout for r in pd.read_sql("select race_id,combo,payout from payouts where bet_type='3連複'",c).itertuples()}
scr=set(map(tuple,pd.read_sql("select race_id,car_no from entries where scratched=1",c).values))
V={'本命(1番)':[0],'ずらし(2〜4番)':[1,2,3],'2番だけ':[1],'3番だけ':[2],'4番だけ':[3],'5〜7番':[4,5,6],'1〜4番':[0,1,2,3]}
rows=[]
for (m,rid),g in t.groupby(['method','race_id']):
    cp=combo_probs(g.car_no.astype(int).tolist(),g.p.to_numpy())['3連複']; s=sorted(cp,key=cp.get,reverse=True)
    gr='G3以上' if g.grade.iloc[0] in ('GP','G1','G2','G3') else 'F'
    for k,ix in V.items():
        for i in ix:
            if i>=len(s): continue
            cb=s[i]; r=100 if any((rid,int(x)) in scr for x in cb.split('-')) else P.get((rid,cb),0)
            rows.append((m,k,g.fold.iloc[0],gr,g.race_class.iloc[0],r))
d=pd.DataFrame(rows,columns=['m','v','fold','gr','cls','ret'])
rng=np.random.default_rng(0)
def tab(x,title):
    out=[]
    for (m,v),g in x.groupby(['m','v']):
        f=g.groupby('fold').ret.mean().round(0).astype(int); r=g.ret.to_numpy(); bs=[rng.choice(r,len(r)).mean() for _ in range(300)]
        out.append((m.replace('LightGBM','LGB'),v,len(r),round((r>0).mean()*100,1),round(r.mean(),1),f"{np.percentile(bs,2.5):.0f}〜{np.percentile(bs,97.5):.0f}",'/'.join(map(str,f))))
    print(title); print(pd.DataFrame(out,columns=['方法','買い方','点数','的中%','回収%','95%','5期間']).to_string(index=False)); print()
pd.set_option('display.width',220)
tab(d,"全レース"); tab(d[d.cls!='A級チャレンジ'],"チャレンジ除く"); tab(d[d.gr=='G3以上'],"G3以上")
