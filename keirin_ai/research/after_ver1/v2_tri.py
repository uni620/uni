"""ver2候補: 3連単の中穴（20〜100倍）を、券種どうしのオッズのずれで選ぶ（ver1の考え方の3連単版）。
各3連単の目について: 自分の市場確率 q（1/オッズ）と、2車単×3連複から作った別の確率
p_alt = q_2車単(a>b) × q_3連複({a,b,c}) / Σ_c' q_3連複({a,b,c'}) を比べる。ロジスティック回帰で当たる確率を出す。
学習 〜10/25、確認 10/26〜（確認期間は最後に1回だけ見る）。"""
import sqlite3, numpy as np, pandas as pd
from scipy.optimize import minimize
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite',timeout=60)
CUT='2026-10-26'
od=pd.read_sql("select race_id,bet_type,combo,odds,fetched_at from odds where is_final=1 and bet_type in ('3連単','2車単','3連複') and race_id>='20260918'",c)
od=od.sort_values('fetched_at').drop_duplicates(['race_id','bet_type','combo'],keep='last')
tri_r=set(od[od.bet_type=='3連単'].race_id)
od=od[od.race_id.isin(tri_r)]
scr=set(pd.read_sql("select distinct race_id from entries where scratched=1",c).race_id)
pay=pd.read_sql("select race_id,combo from payouts where bet_type='3連単'",c); WIN=dict(zip(pay.race_id,pay.combo))
nr=pd.read_sql("select race_id,count(*) n from entries group by race_id",c).set_index('race_id').n.to_dict()
rows=[]
for rid,g in od.groupby('race_id'):
    if rid in scr or rid not in WIN: continue
    d={bt:dict(zip(x.combo,1/x.odds)) for bt,x in g.groupby('bet_type')}
    if len(d)<3: continue
    for bt in d:
        s=sum(d[bt].values()); d[bt]={k:v/s for k,v in d[bt].items()}
    ex,trio,tri=d['2車単'],d['3連複'],d['3連単']
    odds=dict(zip(g[g.bet_type=='3連単'].combo,g[g.bet_type=='3連単'].odds))
    cars=sorted({int(x) for k in tri for x in k.split('>')})
    for k,q in tri.items():
        a,b,cc=map(int,k.split('>'))
        tk=lambda x,y,z:'-'.join(map(str,sorted([x,y,z])))
        den=sum(trio.get(tk(a,b,z),0) for z in cars if z not in (a,b))
        palt=ex.get(f"{a}>{b}",np.nan)*trio.get(tk(a,b,cc),np.nan)/den if den>0 else np.nan
        rows.append((rid,k,q,palt,odds[k],float(nr.get(rid,7)==9),float(WIN[rid]==k)))
T=pd.DataFrame(rows,columns=['race_id','combo','q','palt','odds','nine','hit']).dropna()
T['date']=pd.to_datetime(T.race_id.str[:8]).dt.strftime('%Y-%m-%d')
X=np.column_stack([np.log(T.q),np.log(T.palt),np.log(T.odds),T.nine]); y=T.hit.to_numpy(); tr=(T.date<CUT).to_numpy()
mu,sd=X[tr].mean(0),X[tr].std(0)+1e-9; Z=np.column_stack([np.ones(len(X)),(X-mu)/sd])
def f(w):
    z=Z[tr]@w; p=1/(1+np.exp(-z)); return -(y[tr]*np.log(p+1e-12)+(1-y[tr])*np.log(1-p+1e-12)).sum()+0.5*(w[1:]**2).sum(), Z[tr].T@(p-y[tr])+np.r_[0,w[1:]]
w=minimize(f,np.zeros(Z.shape[1]),jac=True,method='L-BFGS-B').x
T['pm']=1/(1+np.exp(-(Z@w))); T['ev']=T.pm*T.odds
print('係数（標準化）: 定数,自分の確率,別の確率(2車単×3連複),オッズ,9車 =',np.round(w,3))
print('学習 レース',T[tr].race_id.nunique(),' 確認 レース',T[~tr].race_id.nunique())
for per,m in (('学習',tr),('確認',~tr)):
    t=T[m]; print(per,'当たり 実際/見込み 全体',round(t.hit.sum()/t.pm.sum(),3),' 中穴だけ',round(t[t.odds.between(20,100)].hit.sum()/t[t.odds.between(20,100)].pm.sum(),3))
rng=np.random.default_rng(0)
def summ(b):
    if len(b)<20: return f"{len(b)}点（少ない）"
    v=b.hit.to_numpy()*b.odds.to_numpy()*100; i=rng.integers(0,len(v),(2000,len(v))); s=v[i].mean(1)
    return f"{len(b)}点 的中{int(b.hit.sum())} 回収率{v.mean():.0f}% [{np.percentile(s,2.5):.0f}〜{np.percentile(s,97.5):.0f}]"
for lo,hi in ((20,100),(20,50),(50,100)):
    t=T[T.odds.between(lo,hi)]; best=t.loc[t.groupby('race_id').ev.idxmax()]
    for th in (1.0,1.1,1.2,1.3,1.5):
        b=best[best.ev>=th]
        print(f"{lo}〜{hi}倍 期待値{th}以上 1レース1点 | 学習 {summ(b[b.date<CUT])} | 確認 {summ(b[b.date>=CUT])}")
T.to_pickle(f'{S}/v2_tri.pkl'); pd.to_pickle(dict(w=w,mu=mu,sd=sd),f'{S}/v2_tri_model.pkl')
