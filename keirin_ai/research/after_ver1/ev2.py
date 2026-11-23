"""予想(pkl)＋払戻(DB)＋人気(pkl) → 券種ごとの本命1点の回収率と、強気見送りの効果。"""
import sys, sqlite3, pandas as pd, numpy as np
sys.path.insert(0,'/home/user/uni/keirin_ai')
from keirin.combos import combo_probs
pred, dbp, popp, tag = sys.argv[1:5]
t=pd.read_pickle(pred); c=sqlite3.connect(dbp)
pay=pd.read_sql("select race_id,bet_type,combo,payout from payouts",c)
P={(r.race_id,r.bet_type,r.combo):r.payout for r in pay.itertuples()}
hasw=set(pay[pay.bet_type=="ワイド"].race_id)
scr=set(map(tuple,pd.read_sql("select race_id,car_no from entries where scratched=1",c).values))
pop=pd.read_pickle(popp).set_index("race_id")
rows=[]
for (m,rid),g in t.groupby(['method','race_id']):
    if rid not in hasw: continue
    cars=g.car_no.astype(int).tolist(); p=g.p.to_numpy(); cp=combo_probs(cars,p)
    i=int(np.argmax(p)); f1=pop.f1.get(rid); s1=pop.sup1.get(rid)
    strong = (f1==f1 and s1==s1 and int(cars[i])==int(f1) and p[i]>s1)
    for bt in ['2車複','2車単','3連複','ワイド']:
        cb=max(cp[bt],key=cp[bt].get)
        r=100 if any((rid,x) in scr for x in map(int,cb.replace('>','-').split('-'))) else P.get((rid,bt,cb),0)
        rows.append((m,bt,g.fold.iloc[0],g.date.iloc[0],g.race_class.iloc[0],r,cp[bt][cb],rid,len(cars),strong,f1==f1))
d=pd.DataFrame(rows,columns=['method','券種','fold','date','cls','ret','prob','race_id','nr','強気','人気あり'])
d.to_pickle(f"/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad/ev2_{tag}.pkl")
rng=np.random.default_rng(0)
def ci(r):
    if len(r)==0: return "-"
    b=[rng.choice(r,len(r)).mean() for _ in range(500)]; return f"{r.mean():.1f} [{np.percentile(b,2.5):.0f}-{np.percentile(b,97.5):.0f}] n={len(r)}"
out=[]
for (m,bt),g in d.groupby(['method','券種']):
    nc=g[g.cls!='A級チャレンジ']
    out.append({"方法":m,"券種":bt,"全部":ci(g.ret.values),"チャレ除く":ci(nc.ret.values),
                "強気のとき":ci(nc[nc.強気].ret.values),"強気見送り":ci(nc[nc.人気あり&~nc.強気].ret.values),
                "的中率%":round((g.ret>0).mean()*100,1)})
pd.set_option('display.width',300); print(tag, d.race_id.nunique(),"R", d.date.min(), d.date.max()); print(pd.DataFrame(out).to_string())
