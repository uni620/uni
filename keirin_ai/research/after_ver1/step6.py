"""ver1のオッズモデルに「ラインの先頭かどうか」を足す（手順5と同じ分け方: 学習 〜8/31 / 選択 9/1〜10/10 / 最終 10/11〜）。
組み合わせごとの特徴: 先頭の人数（その目に入っている先頭の数）、2車単は1着が先頭か・番手か。"""
import numpy as np, pandas as pd, sqlite3
from scipy.optimize import minimize
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
T1,T2='2026-09-01','2026-10-11'
T=pd.read_pickle(f'{S}/step5_T.pkl')
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite',timeout=60)
E=pd.read_sql("select race_id,car_no,line_pos,line_size from entries",c)
E=E[E.race_id.isin(set(T.race_id))]
role={(r,int(cn)):('先頭' if (ls or 1)>=2 and lp==1 else ('番手' if (ls or 1)>=2 and lp==2 else 'その他')) for r,cn,lp,ls in E.itertuples(index=False)}
cars=T.combo.str.replace('>','-').str.split('-')
T['n_lead']=[sum(role.get((r,int(x)))=='先頭' for x in cs) for r,cs in zip(T.race_id,cars)]
first=[int(cs[0]) for cs in cars]
T['first_lead']=[float(b=='2車単' and role.get((r,f))=='先頭') for r,f,b in zip(T.race_id,first,T.bt)]
T['first_ban']=[float(b=='2車単' and role.get((r,f))=='番手') for r,f,b in zip(T.race_id,first,T.bt)]
COLS={'2車複':['q','q_ex','q_tr','q_qu','q_wi'],'ワイド':['q','q_ex','q_tr','q_qu','q_wi'],'2車単':['q','q_ex','q_tr','q_qu','q_wi','share'],'3連複':['q','s_wi','s_qu','s_ex','w3']}
class LR:
    def fit(self,X,y,C=1.0):
        self.mu=X.mean(0); self.sd=X.std(0)+1e-9; Z=np.column_stack([np.ones(len(X)),(X-self.mu)/self.sd])
        def f(w):
            z=Z@w; p=1/(1+np.exp(-z)); l=-(y*np.log(p+1e-12)+(1-y)*np.log(1-p+1e-12)).sum()+0.5/C*(w[1:]**2).sum()
            g=Z.T@(p-y); g[1:]+=w[1:]/C; return l,g
        self.w=minimize(f,np.zeros(Z.shape[1]),jac=True,method='L-BFGS-B').x; return self
    def p(self,X): return 1/(1+np.exp(-(np.column_stack([np.ones(len(X)),(X-self.mu)/self.sd])@self.w)))
res={}
for mname,extra in (('オッズだけ',False),('オッズ+先頭',True)):
    parts=[]
    for bt,cols in COLS.items():
        x=T[T.bt==bt].dropna(subset=cols).copy()
        F=[np.log(x[cc].astype(float).clip(lower=1e-9)) for cc in cols]+[np.log(x.odds.astype(float)),x.nine.astype(float)]
        if extra: F+=[x.n_lead.astype(float)]+([x.first_lead,x.first_ban] if bt=='2車単' else [])
        F=np.column_stack(F); y=(x.pay>0).astype(float).to_numpy(); tr=(x.date<T1).to_numpy()
        m=LR().fit(F[tr],y[tr]); x['pm']=m.p(F); x['ev']=x.pm*x.odds
        if extra: print(bt,'先頭まわりの係数(標準化)',np.round(m.w[len(cols)+3:],3))
        parts.append(x[['race_id','date','bt','combo','odds','pay','pm','ev']])
    res[mname]=pd.concat(parts)
rng=np.random.default_rng(0)
def summ(b):
    if len(b)<20: return f"{len(b)}点"
    v=b.pay.to_numpy(float); i=rng.integers(0,len(v),(2000,len(v))); s=v[i].mean(1)
    return f"{len(b)}点 {v.mean():.0f}% [{np.percentile(s,2.5):.0f}〜{np.percentile(s,97.5):.0f}]"
for mname,x in res.items():
    print('==',mname)
    for per,(a,b) in (('学習',('0',T1)),('選択',(T1,T2)),('最終',(T2,'9'))):
        xp=x[(x.date>=a)&(x.date<b)]
        out=[f"当たり 実際/見込み {(xp.pay>0).sum()/xp.pm.sum():.3f}"]
        for cap,th in ((10,1.1),(10,1.2),(20,1.2)):
            t=xp[xp.odds<=cap]; best=t.loc[t.groupby('race_id').ev.idxmax()]; out.append(f"{cap}倍・{th}: {summ(best[best.ev>=th])}")
        print(f"  {per}: "+" | ".join(out))
