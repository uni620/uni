"""手順5: オッズのモデル（AIなし / AIのライン連動確率plあり）を学習期間で作り、1レース1点のルールを選択期間で選び、最終確認期間で1回だけ確かめる。"""
import numpy as np, pandas as pd, sys, pickle
from scipy.optimize import minimize
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
T1,T2='2026-09-01','2026-10-11'
T=pd.read_pickle(f'{S}/step5_T.pkl')
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
for mname,extra in (('オッズだけ',[]),('オッズ+AI',['pl'])):
    parts=[]
    for bt,cols in COLS.items():
        x=T[T.bt==bt].copy(); cc=cols+extra; x=x.dropna(subset=cc)
        F=np.column_stack([np.log(x[c].astype(float).clip(lower=1e-9)) for c in cc]+[np.log(x.odds.astype(float)),x.nine.astype(float)])
        y=(x.pay>0).astype(float).to_numpy(); tr=(x.date<T1).to_numpy()
        m=LR().fit(F[tr],y[tr]); x['pm']=m.p(F); x['ev']=x.pm*x.odds
        if extra: print(bt,'AIの係数(標準化)',round(m.w[-3],3),flush=True)
        parts.append(x[['race_id','date','hour','bt','combo','odds','pay','pm','ev']])
    res[mname]=pd.concat(parts)
pickle.dump(res,open(f'{S}/step5_scored.pkl','wb'))
rng=np.random.default_rng(0)
def ci(v):
    idx=rng.integers(0,len(v),(2000,len(v))); s=v[idx].mean(1); return np.percentile(s,[2.5,97.5])
rows=[]
for mname,x in res.items():
    for per,(a,b) in (('学習',('0',T1)),('選択',(T1,T2)),('最終',(T2,'9'))):
        xp=x[(x.date>=a)&(x.date<b)]
        print(mname,per,'当たり 実際/見込み',round((xp.pay>0).sum()/xp.pm.sum(),3))
        for seg,cond in (('全部',xp.hour>=0),('夜とミッドナイト(16時〜)',xp.hour>=16),('朝昼(〜16時)',xp.hour<16)):
            xs=xp[cond]
            for cap in (10,20,30):
                t=xs[xs.odds<=cap]; best=t.loc[t.groupby('race_id').ev.idxmax()]
                for th in (1.0,1.1,1.2,1.3,1.5):
                    bb=best[best.ev>=th]
                    if len(bb)<30: continue
                    v=bb.pay.to_numpy(float); lo,hi=ci(v)
                    rows.append(dict(モデル=mname,期間=per,レース=seg,上限=cap,基準=th,点=len(bb),的中率=round((v>0).mean()*100,1),回収率=round(v.mean(),1),下限=round(lo),上限CI=round(hi)))
Rr=pd.DataFrame(rows); Rr.to_csv(f'{S}/step5_rules.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
print(Rr[Rr.期間=='選択'].sort_values('回収率',ascending=False).head(30).to_string(index=False))
