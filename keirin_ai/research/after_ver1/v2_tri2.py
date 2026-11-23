"""ver2候補（続き）: 3連単の市場確率に、見つけた偏り（展開の特徴）を足して当たる確率を出す。学習 〜10/25、確認 10/26〜。"""
import sqlite3, numpy as np, pandas as pd
from scipy.optimize import minimize
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite',timeout=60)
CUT='2026-10-26'
T=pd.read_pickle(f'{S}/v2_tri.pkl')
E=pd.read_sql("select race_id,car_no,score,line_id,line_pos,line_size from entries where race_id in (select distinct race_id from odds where bet_type='3連単')",c)
E['score']=pd.to_numeric(E.score,errors='coerce')
info={}
for rid,g in E.groupby('race_id'):
    g=g.set_index('car_no'); pos={}; line={}
    for car,r in g.iterrows():
        sz=r.line_size if pd.notna(r.line_size) else 1
        pos[car]='単騎' if sz<=1 else ('先頭' if r.line_pos==1 else ('番手' if r.line_pos==2 else '3番手'))
        line[car]=r.line_id if sz>1 else -car
    s1=set()
    for lid,m in g[g.line_size.fillna(1)>=2].groupby('line_id'):
        m=m.sort_values('line_pos')
        if len(m)>=2 and m.score.iloc[1]-m.score.iloc[0]>=3: s1|=set(m.index)
    ld=g[(g.line_size.fillna(1)>=2)&(g.line_pos==1)].score.sort_values(ascending=False)
    s3=set([ld.index[1]]) if len(ld)>=2 and ld.iloc[0]-ld.iloc[1]<2 else set()
    info[rid]=(pos,line,s1,s3)
F=[]
for rid,k in zip(T.race_id,T.combo):
    a,b,cc=map(int,k.split('>')); pos,line,s1,s3=info.get(rid,({},{},set(),set()))
    F.append([pos.get(a)=='先頭',pos.get(a)=='番手',pos.get(a)=='単騎',line.get(a)==line.get(b),line.get(a)==line.get(b)==line.get(cc),
              bool(s1) and a not in s1, a in s3])
F=np.array(F,float); names=['頭が先頭','頭が番手','頭が単騎','1-2着が同ライン','1-2-3着が同ライン','S1格上番手ライン以外が頭','S3点差小の逆転が頭']
X=np.column_stack([np.log(T.q),np.log(T.odds),T.nine,F]); y=T.hit.to_numpy(); tr=(T.date<CUT).to_numpy()
mu,sd=X[tr].mean(0),X[tr].std(0)+1e-9; Z=np.column_stack([np.ones(len(X)),(X-mu)/sd])
def f(w):
    z=Z[tr]@w; p=1/(1+np.exp(-z)); return -(y[tr]*np.log(p+1e-12)+(1-y[tr])*np.log(1-p+1e-12)).sum()+0.5*(w[1:]**2).sum(), Z[tr].T@(p-y[tr])+np.r_[0,w[1:]]
w=minimize(f,np.zeros(Z.shape[1]),jac=True,method='L-BFGS-B').x
print('係数（標準化）:',dict(zip(['定数','自分の確率','オッズ','9車']+names,np.round(w,3))))
T['pm']=1/(1+np.exp(-(Z@w))); T['ev']=T.pm*T.odds
print('EVの最大値 学習',round(T[tr].ev.max(),2),'確認',round(T[~tr].ev.max(),2),' EV1以上の目の数',(T.ev>=1).sum())
rng=np.random.default_rng(0)
def summ(b):
    if len(b)<20: return f"{len(b)}点（少ない）"
    v=b.hit.to_numpy()*b.odds.to_numpy()*100; i=rng.integers(0,len(v),(2000,len(v))); s=v[i].mean(1)
    return f"{len(b)}点 的中{int(b.hit.sum())} 回収率{v.mean():.0f}% [{np.percentile(s,2.5):.0f}〜{np.percentile(s,97.5):.0f}]"
t=T[T.odds.between(20,100)]; best=t.loc[t.groupby('race_id').ev.idxmax()]
for th in (0.8,0.85,0.9,0.95,1.0,1.1):
    b=best[best.ev>=th]; print(f"20〜100倍 期待値{th}以上 1レース1点 | 学習 {summ(b[b.date<CUT])} | 確認 {summ(b[b.date>=CUT])}")
for qq in (0.9,0.95,0.98):
    thr=best[best.date<CUT].ev.quantile(qq); b=best[best.ev>=thr]
    print(f"学習で上位{int((1-qq)*100)}%の期待値（{thr:.2f}以上） | 学習 {summ(b[b.date<CUT])} | 確認 {summ(b[b.date>=CUT])}")
pd.to_pickle(dict(w=w,mu=mu,sd=sd,cols=['log_q','log_odds','nine']+names,trained='2026-09-18〜2026-10-25',rule='3連単 20〜100倍 期待値1.0以上 1レース1点'),'/home/user/uni/keirin_ai/data/market_models/ana_v1.pkl')
print('saved')
