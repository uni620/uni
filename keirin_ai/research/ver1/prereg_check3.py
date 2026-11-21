import sys
"""事前登録ルールの確認検証（ルール・モデルは固定。prereg_0720.md）。使い方: python prereg_check.py 2026-07-01 2026-09-08"""
import sys, sqlite3, numpy as np, pandas as pd
A,B=sys.argv[1],sys.argv[2]
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
od=pd.read_sql("select o.race_id,r.date,r.n_riders nr,o.bet_type bt,o.combo,o.odds from odds o join races r using(race_id) where o.is_final=1 and r.date between ? and ?",c,params=(A,B))
scr=set(pd.read_sql("select race_id from entries where scratched=1",c).race_id)
pay=pd.read_sql("select race_id,bet_type bt,combo,payout ret from payouts",c)
od=od[~od.race_id.isin(scr)]
full=od.groupby('race_id').bt.nunique(); od=od[od.race_id.isin(full[full==4].index)]
od=od[od.race_id.isin(set(pay[pay.bt=='ワイド'].race_id))]
d=od.merge(pay,on=['race_id','bt','combo'],how='left').fillna({'ret':0})
d['q0']=1/d.odds; d['q']=d.q0/d.groupby(['race_id','bt']).q0.transform('sum')
MOD0=pd.read_pickle('cross3_model_2026-11-20.pkl')
pair=lambda s:'-'.join(sorted(s.replace('>','-').split('-'),key=int))
ex=d[d.bt=='2車単'].copy(); ex['pair']=ex.combo.map(pair); qex=ex.groupby(['race_id','pair']).q.sum().rename('q_ex')
tr=d[d.bt=='3連複']; rows=[]
for rid,cb,q in zip(tr.race_id,tr.combo,tr.q):
    a,b,c=cb.split('-'); rows+=[(rid,f"{a}-{b}",q),(rid,f"{a}-{c}",q),(rid,f"{b}-{c}",q)]
qtr=pd.DataFrame(rows,columns=['race_id','pair','q_tr']).groupby(['race_id','pair']).q_tr.sum()
qqu=d[d.bt=='2車複'].set_index(['race_id','combo']).q.rename('q_qu'); qqu.index.names=['race_id','pair']
qwi=d[d.bt=='ワイド'].set_index(['race_id','combo']).q.rename('q_wi'); qwi.index.names=['race_id','pair']
M=pd.concat([qex,qtr,qqu,qwi],axis=1)
# 1着の確率（2車単の市場から）→ ハービルで3連複の確率
ex['first']=ex.combo.str.split('>').str[0]
pw=ex.groupby(['race_id','first']).q.sum()
res=[]
for bt in ['2車複','ワイド','2車単','3連複']:
    x=d[d.bt==bt].copy()
    if bt!='3連複':
        x['pair']=x.combo.map(pair); x=x.join(M,on=['race_id','pair']); cols=['q','q_ex','q_tr','q_qu','q_wi']
        if bt=='2車単':
            rev=x.set_index(['race_id','combo']).q
            x['q_rev']=[rev.get((r,f"{c.split('>')[1]}>{c.split('>')[0]}"),np.nan) for r,c in zip(x.race_id,x.combo)]
            x['share']=x.q/(x.q+x.q_rev); cols=cols+['share']
    else:
        P=M.to_dict('index'); W=pw.to_dict()
        def feats(rid,cb):
            a,b,c=cb.split('-'); ps=[(a,b),(a,c),(b,c)]
            g=lambda k:sum(P.get((rid,f"{u}-{v}"),{}).get(k,np.nan) for u,v in ps)
            w=[W.get((rid,z),np.nan) for z in (a,b,c)]
            return g('q_wi'),g('q_qu'),g('q_ex'),np.prod(w)
        f=np.array([feats(r,cb) for r,cb in zip(x.race_id,x.combo)])
        x['s_wi'],x['s_qu'],x['s_ex'],x['w3']=f.T; cols=['q','s_wi','s_qu','s_ex','w3']
    x=x.dropna(subset=cols)
    F=np.column_stack([np.log(x[c].clip(1e-9)) for c in cols]+[np.log(x.odds),(x.nr==9).astype(float)])
    m=MOD0[bt]; Z=np.column_stack([np.ones(len(F)),(F-m['mu'])/m['sd']]); x['pm']=1/(1+np.exp(-(Z@m['w']))); x['ev']=x.pm*x.odds; t=np.zeros(len(x),bool)
    te=x[~t]; print(bt,'後半 当たり具合', round(te.ret.gt(0).sum()/te.pm.sum(),3), flush=True)
    res.append(x[~t][['race_id','date','bt','combo','odds','ret','pm','ev']])

T=pd.concat(res); NR=T.race_id.nunique(); T.to_pickle("prereg3_T.pkl")
rng=np.random.default_rng(0)
def ci(g):
    r=g.groupby('race_id').agg(s=('ret','size'),r=('ret','sum'))
    if len(r)<5: return f"R={len(r)}"
    s,v=r.s.values*100.,r.r.values.astype(float); idx=rng.integers(0,len(r),(1000,len(r))); b=v[idx].sum(1)/s[idx].sum(1)*100
    return f"{v.sum()/s.sum()*100:6.1f}% [{np.percentile(b,2.5):3.0f}-{np.percentile(b,97.5):3.0f}] 買ったR={len(r)}/{NR} 収支{int(v.sum()-s.sum()):+,}円"
g=T[T.odds<=30]; g=g.loc[g.groupby('race_id').ev.idxmax()]
print('期間', A, B, NR, 'レース')
print('ルールX EV≥1.1:', ci(g[g.ev>=1.1]), g[g.ev>=1.1].bt.value_counts().to_dict())
print('ルールY EV≥1.2:', ci(g[g.ev>=1.2]), g[g.ev>=1.2].bt.value_counts().to_dict())
