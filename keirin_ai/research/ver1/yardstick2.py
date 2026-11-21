"""物差し2: 2着・3着で、人気（確定オッズ）に AI（ライン連動/ハービル）を混ぜたときの上乗せ。
(A) 2着: 実際の1着を与えたとき、2着を当てる確率。人気=2車単オッズの条件付き、AI=ライン連動の条件付き。
(B) 2車単・3連複の組み合わせ全体: P ∝ exp(a·log q + b·log p)。
前半(〜8/31)で重みを決め、後半(9/1〜)で対数損失を比べる。ライン係数も前半だけで決めたもの。"""
import sys, json, sqlite3, numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0,'/home/user/uni/keirin_ai')
from keirin.line_combos import line_combo_probs, relation_features, ordered_probs
from keirin.combos import combo_probs
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
LP=json.load(open(f'{S}/line_fit_0901.json'))['params']
P=pd.read_pickle(f'{S}/wf_all.pkl'); P=P[P.method=='LightGBM+競走得点']
c=sqlite3.connect(f'{S}/sim.sqlite')
od=pd.read_sql("select race_id,bet_type,combo,odds,fetched_at from odds where is_final=1 and bet_type in ('2車単','3連複') and race_id>='20260611'",c)
od=od.sort_values('fetched_at').drop_duplicates(['race_id','bet_type','combo'],keep='last')
OD={k:dict(zip(g.combo,g.odds)) for k,g in od.groupby(['race_id','bet_type'])}
scr=set(pd.read_sql("select distinct race_id from entries where scratched=1",c).race_id)
A=[];B={'2車単':[],'3連複':[]}
for rid,g in P.groupby('race_id'):
    if rid in scr or (rid,'2車単') not in OD or (rid,'3連複') not in OD: continue
    g=g.sort_values('car_no'); f=g.finish.to_numpy()
    if not all((f==k).sum()==1 for k in (1,2,3)): continue
    cars=g.car_no.astype(int).tolist(); p=g.p.to_numpy(); p=p/p.sum(); d=g.date.iloc[0]
    lines={int(r.car_no):(r.line_id,r.line_pos,r.line_size) for r in g.itertuples() if pd.notna(r.line_id)}
    i1,i2,i3=(cars[int(np.argmax(f==k))] for k in (1,2,3))
    lp=line_combo_probs(cars,p,lines,LP); hp=combo_probs(cars,p)
    ex=OD[(rid,'2車単')]
    # (A) 1着を与えたときの2着
    cand=[j for j in cars if j!=i1 and f"{i1}>{j}" in ex]
    if i2 in cand and len(cand)>=3:
        q=np.array([1/ex[f"{i1}>{j}"] for j in cand]); q/=q.sum()
        a=np.array([lp['2車単'].get(f"{i1}>{j}",1e-9) for j in cand]); a/=a.sum()
        h=np.array([hp['2車単'].get(f"{i1}>{j}",1e-9) for j in cand]); h/=h.sum()
        A.append((d,np.log(np.clip(q,1e-9,1)),np.log(np.clip(a,1e-9,1)),np.log(np.clip(h,1e-9,1)),cand.index(i2)))
    for bt,win in (('2車単',f"{i1}>{i2}"),('3連複','-'.join(map(str,sorted([i1,i2,i3]))))):
        o=OD[(rid,bt)]; ks=[k for k in o if o[k]>0]
        if win not in ks: continue
        q=np.array([1/o[k] for k in ks]); q/=q.sum()
        a=np.array([lp[bt].get(k,1e-9) for k in ks]); a/=a.sum()
        h=np.array([hp[bt].get(k,1e-9) for k in ks]); h/=h.sum()
        B[bt].append((d,np.log(np.clip(q,1e-9,1)),np.log(np.clip(a,1e-9,1)),np.log(np.clip(h,1e-9,1)),ks.index(win)))
def nll(w,R,k):
    t=0
    for r in R:
        s=w[0]*r[1]+w[1]*r[k]; m=s.max(); t-=s[r[4]]-m-np.log(np.exp(s-m).sum())
    return t/len(R)
def fit(R,k,free=True):
    if not free: return np.array([minimize(lambda x:nll([x[0],0],R,k),[1.0],method='L-BFGS-B').x[0],0])
    return minimize(nll,[1.0,0.0],args=(R,k),method='L-BFGS-B').x
rng=np.random.default_rng(0)
def report(name,R):
    tr=[r for r in R if r[0]<'2026-09-01']; te=[r for r in R if r[0]>='2026-09-01']
    for k,lab in ((2,'ライン連動'),(3,'ハービル')):
        w=fit(tr,k); wm=fit(tr,k,False)
        bs=[fit([tr[i] for i in rng.integers(0,len(tr),len(tr))],k)[1] for _ in range(20)]
        print(f"{name} / {lab}: 前半{len(tr)}R 後半{len(te)}R  AI比重b={w[1]:.3f} [{np.percentile(bs,2.5):.3f}〜{np.percentile(bs,97.5):.3f}] 人気a={w[0]:.3f} | 後半 人気{nll([1,0],te,k):.4f} 人気調整{nll(wm,te,k):.4f} AIだけ{nll([0,1],te,k):.4f} 混合{nll(w,te,k):.4f} AIの上乗せ{(nll(wm,te,k)-nll(w,te,k))*1000:+.1f}‰",flush=True)
report("2着（1着を与えたとき）",A)
for bt in B: report(bt,B[bt])
# 後半での上乗せが偶然かどうか（レース単位で入れ替えて95%範囲）
def per_race(w,R,k):
    out=[]
    for r in R:
        s=w[0]*r[1]+w[1]*r[k]; m=s.max(); out.append(-(s[r[4]]-m-np.log(np.exp(s-m).sum())))
    return np.array(out)
for name,R in (("2着",A),("2車単",B['2車単']),("3連複",B['3連複'])):
    tr=[r for r in R if r[0]<'2026-09-01']; te=[r for r in R if r[0]>='2026-09-01']
    dlt=per_race(fit(tr,2,False),te,2)-per_race(fit(tr,2),te,2)
    bs=[dlt[rng.integers(0,len(dlt),len(dlt))].mean()*1000 for _ in range(2000)]
    print(f"{name} ライン連動 後半の上乗せ {dlt.mean()*1000:+.2f}‰ 95%範囲 [{np.percentile(bs,2.5):+.2f}〜{np.percentile(bs,97.5):+.2f}]")
