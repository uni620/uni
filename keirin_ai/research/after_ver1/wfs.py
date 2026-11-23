"""保存した予想（wf_*.pkl）に、いろいろな買い方を当てて回収率を出す。"""
import sys, sqlite3, pandas as pd, numpy as np
sys.path.insert(0,'/home/user/uni/keirin_ai')
from keirin.combos import combo_probs
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
t=pd.read_pickle(sys.argv[1])
c=sqlite3.connect(f'{S}/wf.sqlite')
pay=pd.read_sql("select race_id,bet_type,combo,payout from payouts",c)
P={(r.race_id,r.bet_type,r.combo):r.payout for r in pay.itertuples()}
scr=set(map(tuple,pd.read_sql("select race_id,car_no from entries where scratched=1",c).values))
rows=[]
for (m,rid),g in t.groupby(['method','race_id']):
    cars=g.car_no.astype(int).tolist(); cp=combo_probs(cars,g.p.to_numpy())
    cls=g.race_class.iloc[0]; fold=g.fold.iloc[0]
    def add(name,bt,combos):
        for cb in combos:
            r=100 if any((rid,x) in scr for x in map(int,cb.replace('>','-').split('-'))) else P.get((rid,bt,cb),0)
            rows.append((m,name,bt,fold,cls,r,cp[bt][cb],rid,g.grade.iloc[0],len(cars)))
    for bt in ['2車単','2車複','3連複']:
        s=sorted(cp[bt],key=cp[bt].get,reverse=True)
        add('本命',bt,s[:1])
        if bt=='3連複': add('ずらし',bt,s[1:4])
d=pd.DataFrame(rows,columns=['method','買い方','券種','fold','cls','ret','prob','race_id','grade','nr'])
d.to_pickle(sys.argv[2])
d['チャ']=d.cls=='A級チャレンジ'
def tab(x):
    o=x.groupby(['method','買い方','券種','fold']).ret.mean().unstack('fold').round(0)
    o['全体']=x.groupby(['method','買い方','券種']).ret.mean().round(1); o['点数']=x.groupby(['method','買い方','券種']).size()
    return o
pd.set_option('display.width',250)
print("全レース"); print(tab(d).to_string())
print("\nチャレンジ除く"); print(tab(d[~d['チャ']]).to_string())
