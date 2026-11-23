import sqlite3, numpy as np, pandas as pd
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite',timeout=60)
bn=pd.read_pickle(f'{S}/s5_bn.pkl'); x=bn[(bn['rank']>=4)&bn.玉砕型]
E=pd.read_sql("select e.race_id,e.car_no,e.score,e.line_id,e.line_pos,e.line_size from entries e where race_id in (%s)"%','.join(f"'{r}'" for r in x.race_id.unique()),c)
pay=pd.read_sql("select race_id,combo,payout from payouts where bet_type='3連単'",c); WIN=dict(zip(pay.race_id,pay.combo)); PV=dict(zip(pay.race_id,pay.payout))
P=pd.read_pickle(f'{S}/wf_art.pkl'); P=P[P.method=='+記事の要素']; PD={k:dict(zip(g.car_no.astype(int),g.p)) for k,g in P.groupby('race_id')}
G={k:g for k,g in E.groupby('race_id')}
rows=[]
for r in x.itertuples():
    rid=r.race_id
    if rid not in WIN: continue
    g=G[rid].copy(); g['score']=pd.to_numeric(g.score,errors='coerce'); h=int(r.car_no)
    g['lk']=g.line_id.fillna(-g.car_no).astype(int); lr=g.groupby('lk').score.mean().rank(ascending=False,method='min')[int(r.line_id)]
    L=int(g[(g.line_id==r.line_id)&(g.line_pos==1)].car_no.iloc[0])
    rest=g[g.car_no!=h]
    for way,key in (('AI上位4人で2・3着（12点）','ai'),('競走得点の上位4人で2・3着（12点）','sc')):
        if key=='ai':
            if rid not in PD: continue
            cand=sorted([int(v) for v in rest.car_no],key=lambda v:-PD[rid].get(v,0))[:4]
        else: cand=[int(v) for v in rest.sort_values('score',ascending=False).car_no[:4]]
        cs=[f"{h}>{a}>{b}" for a in cand for b in cand if a!=b]
        rows.append(dict(way=way,date=r.date,lr=lr,pts=len(cs),pay=PV[rid] if WIN[rid] in cs else 0))
X=pd.DataFrame(rows); rng=np.random.default_rng(0)
def summ(d):
    if len(d)<20: return f"{len(d)}R（少ない）"
    st=d.pts.to_numpy()*100.; py=d.pay.to_numpy(); i=rng.integers(0,len(d),(1000,len(d))); b=py[i].sum(1)/st[i].sum(1)*100
    return f"{len(d)}R 的中{(d.pay>0).sum()} 回収率{py.sum()/st.sum()*100:.0f}% [{np.percentile(b,2.5):.0f}〜{np.percentile(b,97.5):.0f}]"
for way in X.way.unique():
    d=X[X.way==way]; print('==',way)
    for lab,m in (('全部',d.lr>0),('ライン強さ1〜2位',d.lr<=2),('ライン強さ3位以下',d.lr>=3)):
        dd=d[m]; print(f"   {lab}: 全期間 {summ(dd)} | 6/11〜8/31 {summ(dd[dd.date<'2026-09-01'])} | 9/1〜 {summ(dd[dd.date>='2026-09-01'])}")
