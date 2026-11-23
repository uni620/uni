"""手順4: レースの種類ごとに、人気にAI（ライン連動・基本7）を混ぜたときの比重bを測る。期間は 6/11〜10/10（最終確認期間は使わない）。"""
import pickle, sqlite3, numpy as np, pandas as pd, sys
from scipy.optimize import minimize
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'; sys.path.insert(0,S)
from step3_fit import combos, R, T2
cols,w2,w3=pickle.load(open(f'{S}/step3_w.pkl','rb'))['基本7']
c=sqlite3.connect(f'{S}/sim.sqlite')
od=pd.read_sql("select race_id,bet_type,combo,odds,fetched_at from odds where is_final=1 and bet_type in ('2車単','3連複') and race_id>='20260611' and race_id<'20261011'",c)
od=od.sort_values('fetched_at').drop_duplicates(['race_id','bet_type','combo'],keep='last')
OD={k:dict(zip(g.combo,g.odds)) for k,g in od.groupby(['race_id','bet_type'])}
meta=pd.read_sql("select race_id,race_class,grade,start_time from races",c).set_index('race_id')
def seg(r):
    m=meta.loc[r['rid']]; st=str(m.start_time or '00:00'); h=int(st[:2])
    tb='朝(〜11時)' if h<11 else '昼(11〜16時)' if h<16 else '夜(16〜20時)' if h<20 else 'ミッドナイト(20時〜)'
    cl=str(m.race_class); cls='S級' if 'S' in cl else 'チャレンジ' if 'チャレンジ' in cl else 'A級'
    return {'時間帯':tb,'級':cls,'グレード':str(m.grade),'車立て':f"{r['n']}車",'ラインの数':'3分戦以下' if r['three'] else '4分戦以上'}
L=[]
for r in R:
    if not (r['date']<T2) or min(r['top'])<0 or (r['rid'],'3連複') not in OD: continue
    i,j,k=r['top']; cars=r['cars']; win='-'.join(map(str,sorted([cars[i],cars[j],cars[k]])))
    o=OD[(r['rid'],'3連複')]; ks=[x for x in o if o[x]>0]
    if win not in ks: continue
    pr,_=combos(r,w2,w3,cols)
    q=np.array([1/o[x] for x in ks]); q/=q.sum(); a=np.array([max(pr['3連複'].get(x,1e-9),1e-9) for x in ks]); a/=a.sum()
    L.append((seg(r),np.log(q),np.log(a),ks.index(win),o[win]))
pickle.dump(L,open(f'{S}/step4_L.pkl','wb'))
def per(w,X): 
    out=[]
    for _,lq,la,y,_ in X:
        s=w[0]*lq+w[1]*la; m=s.max(); out.append(-(s[y]-m-np.log(np.exp(s-m).sum())))
    return np.array(out)
fit=lambda X:minimize(lambda w:per(w,X).mean(),[1.0,0.0],method='L-BFGS-B').x
rng=np.random.default_rng(0)
print('全体',len(L),'R b=',np.round(fit(L),3))
for key in ['時間帯','級','グレード','車立て','ラインの数']:
    for v in sorted({x[0][key] for x in L}):
        X=[x for x in L if x[0][key]==v]
        if len(X)<300: continue
        w=fit(X); bs=[fit([X[i] for i in rng.integers(0,len(X),len(X))])[1] for _ in range(20)]
        print(f"{key} {v}: {len(X)}R  人気a={w[0]:.3f} AI比重b={w[1]:.3f} [{np.percentile(bs,2.5):.3f}〜{np.percentile(bs,97.5):.3f}]",flush=True)
