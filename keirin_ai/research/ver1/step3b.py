"""記事の「97点の先頭+103点の番手は共倒れ / 103点の先頭+97点の番手はちぎれる」を2・3着の決め方に足して測る。"""
import pickle, numpy as np, pandas as pd, sys
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'; sys.path.insert(0,S)
import step3_fit as m
F=pd.read_pickle(f'{S}/feats_0601.pkl')[['race_id','car_no','score']]; F['score']=pd.to_numeric(F.score,errors='coerce')
SC={k:dict(zip(g.car_no.astype(int),g.score)) for k,g in F.groupby('race_id')}
for r in m.R:
    sc=np.array([SC.get(r['rid'],{}).get(c,np.nan) for c in r['cars']]); sc=np.nan_to_num(sc-np.nanmean(sc)) if np.isfinite(sc).any() else np.zeros(r['n'])
    gap=(sc[None,:]-sc[:,None])/5.0                       # j の点数 − i の点数（5点単位）
    rel=r['X2'][...,1:4]
    add2=np.stack([rel[...,0]*gap, rel[...,1]*gap],-1)    # 真後ろ×差 / 真ん前×差
    r['X2']=np.concatenate([r['X2'],add2],-1)
    rr=r['X3'][...,1:4]
    g3=np.maximum(gap[:,None,:],gap[None,:,:])
    add3=np.stack([rr[...,0]*g3, rr[...,1]*g3],-1)
    r['X3']=np.concatenate([r['X3'],add3],-1)
tr=[r for r in m.R if r['date']<m.T1]
cols=list(range(7))+[14,15]
w2=m.fitw(tr,2,cols); w3=m.fitw(tr,3,cols); print('2着',np.round(w2,3),'3着',np.round(w3,3))
W=pickle.load(open(f'{S}/step3_w.pkl','rb')); W['+点数差(記事)']=(cols,w2,w3)
W={k:W[k] for k in ('基本7','+点数差(記事)')}
pickle.dump(W,open(f'{S}/step3_w2.pkl','wb')); pickle.dump(m.R,open(f'{S}/step3_R2.pkl','wb'))
