import pandas as pd,numpy as np
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
E=pd.read_pickle(f'{S}/hx2.pkl')[['race_id','date','car_no','finish','lkey','line_pos','ライン人数','先頭','番手','単騎','3番手+','score','自分のラインのpB','pB']]
E=E[(E.date>='2026-09-01')]
W=E[E.finish==1].set_index('race_id')
X=E[E.finish.isin([2,3])].copy()
X['hp']=X.race_id.map(np.select([W.先頭==1,W.番手==1,W.単騎==1],['先頭','番手','単騎'],'3番手+').__iter__().__class__ and pd.Series(np.select([W.先頭==1,W.番手==1,W.単騎==1],['先頭','番手','単騎'],'3番手+'),index=W.index))
same=X.lkey==X.race_id.map(W.lkey); d=X.line_pos-X.race_id.map(W.line_pos)
X['どこから']=np.select([same&(d==1),same&(d==-1),same&(d>=2),same&(d<=-2),X.単騎==1,X.先頭==1,X.番手==1],
   ['同ライン 直後','同ライン 直前','同ライン 2つ後ろ','同ライン 2つ前','他の単騎','他ラインの先頭','他ラインの番手'],'他ラインの3番手+')
for k in (2,3):
    t=pd.crosstab(X[X.finish==k].hp,X[X.finish==k].どこから,normalize='index').mul(100).round(1)
    print(f'--- {k}着はどこから来るか（頭の位置別, %）'); print(t.to_string())
# 頭の位置別に「同ラインの直後がいる」ときの2着率、頭のラインの主導権確率(pB)で分ける
x2=X[X.finish==2]; W['lpB']=W.自分のラインのpB
x2['頭ラインpB']=x2.race_id.map(W.lpB)
x2['頭の後ろが2着']=(x2.どこから=='同ライン 直後')
r=x2[x2.hp.isin(['先頭','番手'])&(x2.race_id.map(W.ライン人数)>=2)]
print(r.groupby(['hp',pd.cut(r.頭ラインpB,[0,.2,.4,.6,1])],observed=True).頭の後ろが2着.agg(['size','mean']).round(3))
