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
pair=lambda s:'-'.join(sorted(s.replace('>','-').split('-'),key=int))
ex=d[d.bt=='2車単'].copy(); ex['pair']=ex.combo.map(pair); qex=ex.groupby(['race_id','pair']).q.sum().rename('q_ex')
tr=d[d.bt=='3連複']; rows=[]
for rid,cb,q in zip(tr.race_id,tr.combo,tr.q):
    a,b,cc=cb.split('-'); rows+=[(rid,f"{a}-{b}",q),(rid,f"{a}-{cc}",q),(rid,f"{b}-{cc}",q)]
qtr=pd.DataFrame(rows,columns=['race_id','pair','q_tr']).groupby(['race_id','pair']).q_tr.sum()
qqu=d[d.bt=='2車複'].set_index(['race_id','combo']).q.rename('q_qu'); qqu.index.names=['race_id','pair']
qwi=d[d.bt=='ワイド'].set_index(['race_id','combo']).q.rename('q_wi'); qwi.index.names=['race_id','pair']
M=pd.concat([qex,qtr,qqu,qwi],axis=1)
mod=pd.read_pickle('cross2_model_2026-11-20.pkl')
rng=np.random.default_rng(0)
def ci(g):
    r=g.groupby('race_id').agg(s=('ret','size'),r=('ret','sum'))
    if len(r)<5: return f"R={len(r)}"
    s,v=r.s.values*100.,r.r.values.astype(float); idx=rng.integers(0,len(r),(1000,len(r))); b=v[idx].sum(1)/s[idx].sum(1)*100
    return f"{v.sum()/s.sum()*100:6.1f}% [{np.percentile(b,2.5):3.0f}-{np.percentile(b,97.5):3.0f}] R={len(r)} 点={int(s.sum()/100)}"
print('対象', d.race_id.nunique(),'レース', d.date.min(), d.date.max())
for bt in ['ワイド','2車複']:
    x=d[d.bt==bt].copy(); x['pair']=x.combo.map(pair); x=x.join(M,on=['race_id','pair'])
    m=mod[bt]; cols=m['cols']; x=x.dropna(subset=cols)
    F=np.column_stack([np.log(x[k].clip(1e-6)) for k in cols]+[np.log(x.odds),(x.nr==9).astype(float)])
    Z=np.column_stack([np.ones(len(F)),(F-m['mu'])/m['sd']]); x['pm']=1/(1+np.exp(-(Z@m['w']))); x['ev']=x.pm*x.odds
    print(f'== {bt} 当たり具合 実際/予想', round(x.ret.gt(0).sum()/x.pm.sum(),3))
    if bt=='ワイド':
        print(' 主ルール EV≥1.0 30倍以下:', ci(x[(x.ev>=1.0)&(x.odds<=30)]))
        print(' 副 EV≥1.05 30倍以下:', ci(x[(x.ev>=1.05)&(x.odds<=30)]))
        x['r']=x.q_tr/3/x.q; print(' 副 割安度≥1.2 30倍以下:', ci(x[(x.r>=1.2)&(x.odds<=30)]))
    else:
        x['r']=x.q_ex/x.q; print(' 副 割安度≥1.2 30倍以下:', ci(x[(x.r>=1.2)&(x.odds<=30)]))
