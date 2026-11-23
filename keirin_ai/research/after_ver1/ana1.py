import pandas as pd, numpy as np
d=pd.read_pickle('odds_val.pkl')
pd.set_option('display.width',250)
w=d[(d.ret>0)&(d.method=='LightGBM')]
print("確定オッズ×100 と払戻の一致率", ((w.odds*100-w.ret).abs()<=w.ret*0.02+10).mean().round(3), "（ワイドは下限）")
m=d[d.method=='LightGBM']
# A) 全部均等買い、オッズ帯別
m=m.assign(band=pd.cut(m.odds,[0,2,5,10,20,50,100,1e9],labels=['~2','2-5','5-10','10-20','20-50','50-100','100~']))
print("\n[A] オッズ帯ごとに全部買ったときの回収率%")
print((m.groupby(['bt','band'],observed=True).ret.sum()/m.groupby(['bt','band'],observed=True).ret.size()).unstack().round(0))
print("\n[B] 1番人気1点買い")
f=m[m.rk_q==1]; print((f.groupby('bt').ret.mean()).round(1).to_dict(), "的中率", (f.groupby('bt').ret.apply(lambda x:(x>0).mean())*100).round(1).to_dict())
