import pandas as pd, numpy as np
d=pd.read_pickle('odds_val.pkl'); d=d[d.method=='LightGBM'][['race_id','date','cls','nr','bt','combo','odds','ret','q0']]
rng=np.random.default_rng(0)
def ci(g):
    r=g.groupby('race_id').agg(s=('ret','size'),r=('ret','sum'))
    if len(r)<5: return f"R={len(r)}"
    s,v=r.s.values*100.,r.r.values.astype(float); idx=rng.integers(0,len(r),(400,len(r))); b=v[idx].sum(1)/s[idx].sum(1)*100
    return f"{v.sum()/s.sum()*100:6.1f}% [{np.percentile(b,2.5):3.0f}-{np.percentile(b,97.5):3.0f}] R={len(r)} 点={int(s.sum()/100)}"
def half(g):
    a=g[g.date<'2026-10-15']; b=g[g.date>='2026-10-15']
    f=lambda h: f"{h.ret.sum()/len(h):.0f}%" if len(h) else '-'
    return f"前半{f(a)} 後半{f(b)}"
# 各券種の市場確率（控除込みで合計1）
d['q']=d.q0/d.groupby(['race_id','bt']).q0.transform('sum')
ex=d[d.bt=='2車単'].copy(); ex['pair']=ex.combo.str.replace('>','-').map(lambda s:'-'.join(sorted(s.split('-'),key=int)))
q_ex=ex.groupby(['race_id','pair']).q.sum()
tr=d[d.bt=='3連複'].copy()
rows=[]
for rid,cb,q in zip(tr.race_id,tr.combo,tr.q):
    a,b,c=cb.split('-')
    for k in (f"{a}-{b}",f"{a}-{c}",f"{b}-{c}"): rows.append((rid,k,q))
q_trw=pd.DataFrame(rows,columns=['race_id','pair','q']).groupby(['race_id','pair']).q.sum()
q_trp=None
qu=d[d.bt=='2車複'].set_index(['race_id','combo']); wi=d[d.bt=='ワイド'].set_index(['race_id','combo'])
qu['q_alt']=q_ex.reindex(qu.index).values; qu['r']=qu.q_alt/qu.q
wi['q_alt']=q_trw.reindex(wi.index).values/3*1.0; wi['q_self']=wi.q
wi['r']=wi.q_alt/wi.q_self
pd.set_option('display.width',250)
for name,t in [('2車複（2車単の人気から見た割安度）',qu),('ワイド（3連複の人気から見た割安度）',wi)]:
    t=t.reset_index(); print('==',name)
    for lo,hi in [(0,0.8),(0.8,0.95),(0.95,1.05),(1.05,1.2),(1.2,1.4),(1.4,9)]:
        g=t[(t.r>=lo)&(t.r<hi)]; print(f"  割安度 {lo}-{hi}: ",ci(g),half(g))
    for th in [1.2,1.4]:
        g=t[(t.r>=th)&(t.odds<=30)]; print(f"  割安度≥{th} かつ30倍以下: ",ci(g),half(g))
