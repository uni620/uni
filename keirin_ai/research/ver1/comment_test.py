import pandas as pd, numpy as np, re
from scipy.optimize import minimize
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
M=pd.read_pickle(f'{S}/honshi_marks.pkl')[['race_id','car_no','comment']]
F=pd.read_pickle(f'{S}/feats_0601.pkl')[['race_id','car_no','date','finish','line_id','line_pos','line_size']]
T=pd.read_pickle(f'{S}/step5_T.pkl'); ex=T[T.bt=='2車単'][['race_id','combo','q']].copy(); ex['car_no']=ex.combo.str.split('>').str[0].astype(int)
W=ex.groupby(['race_id','car_no']).q.sum().reset_index()
D=W.merge(F,on=['race_id','car_no']).merge(M,on=['race_id','car_no'],how='left')
D['won']=(pd.to_numeric(D.finish,errors='coerce')==1).astype(float)
c=D.comment.astype(str)
lead=(D.line_size.fillna(1)>=2)&(D.line_pos==1); solo=D.line_size.fillna(1)<=1
D['x_lead']=lead.astype(float)
D['x_lead_jiriki']=(lead&c.str.fullmatch(r'(自力|自力で|自力です|自分で|自分でやる)。?')).astype(float)
D['x_lead_jizai']=(lead&c.str.contains('自在')).astype(float)
D['x_solo']=solo.astype(float)
D['x_solo_kimezu']=(solo&c.str.contains('決め')).astype(float)
D['x_solo_jizai']=(solo&c.str.contains('自在')).astype(float)
X=['x_lead','x_lead_jiriki','x_lead_jizai','x_solo','x_solo_kimezu','x_solo_jizai']
rng=np.random.default_rng(0)
def ratio(g):
    a=g[['won','q']].to_numpy(); r=a[:,0].sum()/a[:,1].sum(); b=[]
    for _ in range(300):
        i=rng.integers(0,len(a),len(a)); b.append(a[i,0].sum()/a[i,1].sum())
    return f"{len(g)}人 実際/見込み {r:.3f} [{np.percentile(b,2.5):.3f}〜{np.percentile(b,97.5):.3f}]"
for lab,m in (('先頭・「自力」だけ',D.x_lead_jiriki==1),('先頭・「自力自在」',D.x_lead_jizai==1),('先頭・その他',(D.x_lead==1)&(D.x_lead_jiriki==0)&(D.x_lead_jizai==0)),
              ('番手以降',(D.x_lead==0)&(D.x_solo==0)),('単騎・「決めず」',D.x_solo_kimezu==1),('単騎・「自在」',D.x_solo_jizai==1),('単騎・その他',(D.x_solo==1)&(D.x_solo_kimezu==0)&(D.x_solo_jizai==0))):
    print(lab,'| 前半',ratio(D[m&(D.date<'2026-09-01')]),'| 後半',ratio(D[m&(D.date>='2026-09-01')]))
R=[]
for rid,g in D.groupby('race_id'):
    if g.won.sum()!=1: continue
    R.append((g.date.iloc[0],np.log(g.q.clip(1e-6).to_numpy()),g[X].to_numpy(),int(np.argmax(g.won.to_numpy()))))
def per(w,RR):
    o=[]
    for _,lq,x,y in RR:
        s=w[0]*lq+x@w[1:]; m=s.max(); o.append(-(s[y]-m-np.log(np.exp(s-m).sum())))
    return np.array(o)
tr=[r for r in R if r[0]<'2026-09-01']; te=[r for r in R if r[0]>='2026-09-01']
k=len(X)
def fit(mask):
    idx=[0]+[i+1 for i in range(k) if mask[i]]
    def f(v):
        w=np.zeros(k+1); w[idx]=v; return per(w,tr).mean()
    v=minimize(f,np.r_[1.0,np.zeros(len(idx)-1)],method='L-BFGS-B').x; w=np.zeros(k+1); w[idx]=v; return w
wm=fit([0]*k); wp=fit([1,0,0,1,0,0]); wf=fit([1]*k)
print('前半で決めた係数（全部入り）: 人気',round(wf[0],3),dict(zip(X,np.round(wf[1:],3))))
for lab,w in (('位置だけ（先頭か・単騎か）',wp),('位置＋コメント',wf)):
    d=per(wm,te)-per(w,te); bs=[d[rng.integers(0,len(d),len(d))].mean()*1000 for _ in range(1000)]
    print(f'後半（9/1〜、{len(te)}R） {lab} の上乗せ: {d.mean()*1000:+.2f}‰ [{np.percentile(bs,2.5):+.2f}〜{np.percentile(bs,97.5):+.2f}]')
d=per(wp,te)-per(wf,te); bs=[d[rng.integers(0,len(d),len(d))].mean()*1000 for _ in range(1000)]
print(f'コメント分だけの上積み: {d.mean()*1000:+.2f}‰ [{np.percentile(bs,2.5):+.2f}〜{np.percentile(bs,97.5):+.2f}]')
