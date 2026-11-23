from hx_lib import *
Y=pd.read_pickle(f'{S}/hx_Y.pkl'); W=pd.read_pickle(f'{S}/hx_within_pred.pkl')[['pin']]
Y['sm']=np.exp(Y.p-Y.groupby('race_id').p.transform('max')); Y['sm']/=Y.groupby('race_id').sm.transform('sum')
top=Y.loc[Y.groupby('race_id').p.idxmax()].set_index('race_id')
res={'元':top.win.mean()}
# 方式: AI本命のラインが先頭/番手どちらかの2人ラインなら、ライン内AIで先頭か番手を選び直す
pick=[]
for rid,t in top.iterrows():
    lk=t.lkey; g=Y[Y.lkey==lk]
    if lk in W.index and t.先頭+t.番手>0:
        pin=W.loc[lk,'pin']; want=1 if pin>0.5 else 2
        r=g[(g.先頭==1) if want==1 else (g.番手==1)]
        pick.append(r.win.iloc[0] if len(r) else t.win)
    else: pick.append(t.win)
res['AIのライン×ライン内AI']=np.mean(pick)
# 自信の強さで選び直す: ライン内AIの確信度が強いときだけ（|pin-0.5|>0.15）
pick=[]
for rid,t in top.iterrows():
    lk=t.lkey; g=Y[Y.lkey==lk]
    if lk in W.index and t.先頭+t.番手>0 and abs(W.loc[lk,'pin']-0.5)>0.15:
        want=1 if W.loc[lk,'pin']>0.5 else 2; r=g[(g.先頭==1) if want==1 else (g.番手==1)]
        pick.append(r.win.iloc[0] if len(r) else t.win)
    else: pick.append(t.win)
res['…確信が強いときだけ']=np.mean(pick)
print({k:round(v*100,2) for k,v in res.items()}, len(top))
