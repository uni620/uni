import sys, sqlite3, pandas as pd, numpy as np
sys.path.insert(0,'/home/user/uni/keirin_ai')
from keirin.combos import combo_probs, combo_key
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
t=pd.read_pickle(f'{S}/preds_test.pkl')
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
pay=pd.read_sql("select race_id,bet_type,combo,payout from payouts",c)
pay=pay[pay.race_id.isin(t.race_id.unique())]
P={(r.race_id,r.bet_type,r.combo):r.payout for r in pay.itertuples()}
scr=set(map(tuple,pd.read_sql("select race_id,car_no from entries where scratched=1",c).values))
rows=[]
for rid,g in t.groupby('race_id'):
    if not any(k[0]==rid for k in []): pass
    g=g.sort_values('p',ascending=False)
    cars=g.car_no.astype(int).tolist(); pr=g.p.to_numpy()
    cp=combo_probs(cars,pr)
    rk=cars  # sorted by p
    half='前' if g.date.iloc[0]<'2026-11-05' else '後'
    def add(name,bt,combos):
        st=len(combos)*100; ret=0
        for cb in combos:
            if any((rid,x) in scr for x in map(int,cb.replace('>','-').split('-'))): ret+=100
            else: ret+=P.get((rid,bt,cb),0)
        rows.append((name,half,g.race_class.iloc[0],st,ret,int(ret>0)))
    b3=max(cp['3連複'],key=cp['3連複'].get)
    add('3連複 今(1点)','3連複',[b3])
    add('3連複 2頭軸→3着3人(3点)','3連複',[combo_key('3連複',(rk[0],rk[1],x)) for x in rk[2:5]])
    add('3連複 1頭軸→2・3着は2〜5位(6点)','3連複',[combo_key('3連複',(rk[0],)+tuple(cmb)) for cmb in __import__('itertools').combinations(rk[1:5],2)])
    s3=sorted(cp['3連複'],key=cp['3連複'].get,reverse=True)
    add('3連複 確率2〜4番目(3点)','3連複',s3[1:4])
    b2=max(cp['2車複'],key=cp['2車複'].get); add('2車複 今(1点)','2車複',[b2])
    add('2車複 1位軸→2・3位(2点)','2車複',[combo_key('2車複',(rk[0],x)) for x in rk[1:3]])
    b1=max(cp['2車単'],key=cp['2車単'].get); add('2車単 今(1点)','2車単',[b1])
    add('2車単 1位→2・3位(2点)','2車単',[combo_key('2車単',(rk[0],x)) for x in rk[1:3]])
d=pd.DataFrame(rows,columns=['作戦','half','cls','st','ret','hit'])
o=d.groupby(['作戦','half']).agg(st=('st','sum'),ret=('ret','sum'),hit=('hit','mean')).reset_index()
o['回収']=(o.ret/o.st*100).round(1); o['的中']=(o.hit*100).round(1)
print(o.pivot(index='作戦',columns='half',values=['的中','回収']).to_string())
x=d[d.cls!='A級チャレンジ']
o=x.groupby(['作戦','half']).agg(st=('st','sum'),ret=('ret','sum')).reset_index(); o['回収']=(o.ret/o.st*100).round(1)
print("\nA級チャレンジ除く"); print(o.pivot(index='作戦',columns='half',values='回収').to_string())
