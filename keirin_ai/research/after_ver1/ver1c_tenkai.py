"""ver1c（ワイドなし・10倍以下・EV基準）に 展開AI v1/v2・強さAI(ライン連動pl) を混ぜたらどうなるか（確定オッズ）。
学習 9/1〜10/10 → 確認 10/11〜11/20。展開AIは 8/31 までで学習したので両期間とも学習外。"""
import numpy as np, pandas as pd, itertools
from scipy.optimize import minimize
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
T=pd.read_pickle(f'{S}/step5_T.pkl'); T=T[T.date>='2026-09-01'].copy()
W=pd.read_pickle(f'{S}/tenkai.pkl')[['race_id','car_no','p_tenkai']].merge(pd.read_pickle(f'{S}/tenkai2.pkl')[['race_id','car_no','p_v2']],on=['race_id','car_no'])
W=W[W.race_id.isin(T.race_id)]
def combo_probs(cars,p):
    p=np.asarray(p,float); p=p/p.sum(); n=len(p); idx={c:i for i,c in enumerate(cars)}
    out={'2車単':{},'2車複':{},'ワイド':{},'3連複':{}}
    for i,j in itertools.permutations(range(n),2):
        v=p[i]*p[j]/max(1-p[i],1e-9); out['2車単'][f"{cars[i]}>{cars[j]}"]=v
        k=f"{min(cars[i],cars[j])}-{max(cars[i],cars[j])}"; out['2車複'][k]=out['2車複'].get(k,0)+v
    for i,j,k in itertools.permutations(range(n),3):
        v=p[i]*p[j]/max(1-p[i],1e-9)*p[k]/max(1-p[i]-p[j],1e-9)
        s=sorted((cars[i],cars[j],cars[k])); key="-".join(map(str,s)); out['3連複'][key]=out['3連複'].get(key,0)+v
        for a,b in ((s[0],s[1]),(s[0],s[2]),(s[1],s[2])): kk=f"{a}-{b}"; out['ワイド'][kk]=out['ワイド'].get(kk,0)+v
    return out
for col,name in (('p_tenkai','pt1'),('p_v2','pt2')):
    vals=[]
    D={}
    for rid,g in W.groupby('race_id'):
        g=g.sort_values('car_no'); D[rid]=combo_probs(g.car_no.astype(int).tolist(),g[col].fillna(1e-3).values)
    T[name]=[D.get(r,{}).get(b,{}).get(c,np.nan) for r,b,c in zip(T.race_id,T.bt,T.combo)]
    print(name,'ok',T[name].notna().mean(),flush=True)
T.to_pickle(f'{S}/ver1c_T.pkl')
COLS={'2車複':['q','q_ex','q_tr','q_qu','q_wi'],'ワイド':['q','q_ex','q_tr','q_qu','q_wi'],'2車単':['q','q_ex','q_tr','q_qu','q_wi','share'],'3連複':['q','s_wi','s_qu','s_ex','w3']}
class LR:
    def fit(self,X,y,C=1.0):
        self.mu=X.mean(0); self.sd=X.std(0)+1e-9; Z=np.column_stack([np.ones(len(X)),(X-self.mu)/self.sd])
        def f(w):
            z=Z@w; p=1/(1+np.exp(-z)); l=-(y*np.log(p+1e-12)+(1-y)*np.log(1-p+1e-12)).sum()+0.5/C*(w[1:]**2).sum()
            g=Z.T@(p-y); g[1:]+=w[1:]/C; return l,g
        self.w=minimize(f,np.zeros(Z.shape[1]),jac=True,method='L-BFGS-B').x; return self
    def p(self,X): return 1/(1+np.exp(-(np.column_stack([np.ones(len(X)),(X-self.mu)/self.sd])@self.w)))
T1,T2='2026-09-01','2026-10-11'
VARS={'オッズAIだけ(ver1c)':[],'＋展開AI v1':['pt1'],'＋展開AI v2':['pt2'],'＋強さAI(ライン連動)':['pl'],'＋展開v1＋強さ':['pt1','pl']}
need=['pt1','pt2','pl']
rng=np.random.default_rng(0); rows=[]
for vn,extra in VARS.items():
    parts=[]
    for bt,cols in COLS.items():
        if bt=='ワイド': continue
        x=T[T.bt==bt].dropna(subset=cols+need).copy(); cc=cols+extra
        F=np.column_stack([np.log(x[c].astype(float).clip(lower=1e-9)) for c in cc]+[np.log(x.odds.astype(float)),x.nine.astype(float)])
        y=(x.pay>0).astype(float).to_numpy(); tr=(x.date<T2).to_numpy()
        m=LR().fit(F[tr],y[tr]); x['pm']=m.p(F); x['ev']=x.pm*x.odds
        if extra: print(vn,bt,'AI係数',np.round(m.w[len(cols)+1:len(cc)+1],3),flush=True)
        parts.append(x[['race_id','date','bt','combo','odds','pay','pm','ev']])
    x=pd.concat(parts); x=x[x.odds<=10]
    best=x.loc[x.groupby('race_id').ev.idxmax()]
    for per,cond in (('学習9/1〜10/10',best.date<T2),('確認10/11〜11/20',best.date>=T2)):
        b0=best[cond]
        for th in (1.0,1.1,1.2,1.3):
            bb=b0[b0.ev>=th]; v=bb.pay.to_numpy(float)
            if len(v)<10: rows.append((vn,per,th,len(v),'','','')); continue
            s=v[rng.integers(0,len(v),(2000,len(v)))].mean(1)
            rows.append((vn,per,th,len(v),int((v>0).sum()),round(v.mean(),1),f"{np.percentile(s,2.5):.0f}〜{np.percentile(s,97.5):.0f}"))
pd.set_option('display.width',220); pd.set_option('display.max_rows',200)
print(pd.DataFrame(rows,columns=['モデル','期間','EV基準','点','的中','回収率%','95%幅']).to_string(index=False))
