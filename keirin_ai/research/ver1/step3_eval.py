import pickle, sqlite3, numpy as np, pandas as pd, sys
from scipy.optimize import minimize
sys.path.insert(0,S:='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad')
from step3_fit import combos, R, T1, T2
W=pickle.load(open(f'{S}/step3_w.pkl','rb'))
c=sqlite3.connect(f'{S}/sim.sqlite')
od=pd.read_sql("select race_id,bet_type,combo,odds,fetched_at from odds where is_final=1 and bet_type in ('2車単','3連複') and race_id>='20260901'",c)
od=od.sort_values('fetched_at').drop_duplicates(['race_id','bet_type','combo'],keep='last')
OD={k:dict(zip(g.combo,g.odds)) for k,g in od.groupby(['race_id','bet_type'])}
period=sys.argv[1] if len(sys.argv)>1 else 'sel'
rs=[r for r in R if (T1<=r['date']<T2 if period=='sel' else r['date']>=T2) and min(r['top'])>=0]
def rows(r,pr):
    i,j,k=r['top']; cars=r['cars']; out={}
    for bt,win in (('2車単',f"{cars[i]}>{cars[j]}"),('3連複','-'.join(map(str,sorted([cars[i],cars[j],cars[k]]))))):
        o=OD.get((r['rid'],bt),{}); ks=[x for x in o if o[x]>0]
        if win not in ks: continue
        q=np.array([1/o[x] for x in ks]); q/=q.sum(); a=np.array([max(pr[bt].get(x,1e-9),1e-9) for x in ks]); a/=a.sum()
        out[bt]=(np.log(q),np.log(a),ks.index(win))
    ex=OD.get((r['rid'],'2車単'),{}); cand=[x for x in range(r['n']) if x!=i and f"{cars[i]}>{cars[x]}" in ex]
    if j in cand:
        q=np.array([1/ex[f"{cars[i]}>{cars[x]}"] for x in cand]); q/=q.sum()
        a=np.array([pr['2車単'][f"{cars[i]}>{cars[x]}"] for x in cand]); a/=a.sum()
        out['2着']=(np.log(q),np.log(np.clip(a,1e-9,1)),cand.index(j))
    return out
def per(w,L): 
    o=[]
    for lq,la,y in L:
        s=w[0]*lq+w[1]*la; m=s.max(); o.append(-(s[y]-m-np.log(np.exp(s-m).sum())))
    return np.array(o)
def fit(L,free=True):
    if free: return minimize(lambda w:per(w,L).mean(),[1.0,0.0],method='L-BFGS-B').x
    a=minimize(lambda x:per([x[0],0],L).mean(),[1.0],method='L-BFGS-B').x[0]; return np.array([a,0])
rng=np.random.default_rng(0)
store={}
for name,(cols,w2,w3) in W.items():
    Ls={'2着':[],'2車単':[],'3連複':[]}
    for r in rs:
        pr,_=combos(r,w2,w3,cols)
        for k,v in rows(r,pr).items(): Ls[k].append(v)
    for k,L in Ls.items():
        wm=fit(L,False); wb=fit(L) if period=='sel' else None
        if period!='sel': wb=pickle.load(open(f'{S}/step3_blend.pkl','rb'))[(name,k)]
        dl=per(wm,L)-per(wb,L); bs=[dl[rng.integers(0,len(dl),len(dl))].mean()*1000 for _ in range(1000)]
        store[(name,k)]=wb
        print(f"{period} {name:8s} {k:4s} {len(L)}R  AI比重b={wb[1]:.3f}  AIだけ{per([0,1],L).mean():.4f} 人気調整{per(wm,L).mean():.4f}  上乗せ{dl.mean()*1000:+.2f}‰ [{np.percentile(bs,2.5):+.2f}〜{np.percentile(bs,97.5):+.2f}]",flush=True)
if period=='sel': pickle.dump(store,open(f'{S}/step3_blend.pkl','wb'))
