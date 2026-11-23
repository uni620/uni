import sys, json, sqlite3, numpy as np, pandas as pd
sys.path.insert(0,'/home/user/uni/keirin_ai'); sys.path.insert(0,'/home/user/uni/keirin_ai/scripts')
from fit_line_combos import load
from keirin.line_combos import line_combo_probs
from keirin.combos import combo_probs
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
pr=json.load(open(f'{S}/line_fit_1015.json'))['params']
races=[r for r in load(f'{S}/wf_v2.pkl','LightGBM+競走得点') if r['date']>='2026-10-15']
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
pay=pd.read_sql("select race_id,bet_type,combo,payout from payouts",c); P={(a,b,d):e for a,b,d,e in pay.itertuples(index=False)}
rows=[]
for r in races:
    i,j,k=r['ijk']; cars=r['cars']; ans={'2車単':f"{cars[i]}>{cars[j]}",'2車複':'-'.join(map(str,sorted([cars[i],cars[j]]))),'3連複':'-'.join(map(str,sorted([cars[i],cars[j],cars[k]])))}
    for name,cp in [('今まで',combo_probs(cars,r['p'])),('ライン連動',line_combo_probs(cars,r['p'],r['lines'],pr))]:
        for bt in ['2車複','2車単','3連複','ワイド']:
            rk=sorted(cp[bt],key=cp[bt].get,reverse=True)
            if bt!='ワイド': ll=-np.log(max(cp[bt][ans[bt]],1e-9))
            else: ll=np.nan
            for n,cb in enumerate(rk[:4]):
                rows.append((name,bt,r['race_id'],n+1,cb,P.get((r['race_id'],bt,cb),0),ll if n==0 else np.nan))
d=pd.DataFrame(rows,columns=['方法','券種','race_id','順位','combo','ret','ll'])
print('テスト 10/15-11/19', len(races),'レース')
t=d[d.順位==1].groupby(['券種','方法']).agg(外れ具合=('ll','mean'),本命的中率=('ret',lambda x:round((x>0).mean()*100,1)),本命回収率=('ret','mean'))
t['外れ具合']=t.外れ具合.round(3); t['本命回収率']=t.本命回収率.round(1); print(t.to_string())
z=d[(d.券種=='3連複')&(d.順位.between(2,4))].groupby('方法').ret.agg(['mean',lambda x:(x>0).mean()*100]).round(1); print('3連複ずらし(2-4位)',z.to_dict('index'))
z=d[(d.券種=='3連複')&(d.順位<=3)].groupby('方法').ret.agg(['mean']).round(1); print('3連複 上位3点',z.to_dict('index'))
d.to_pickle(f'{S}/line_eval.pkl')
