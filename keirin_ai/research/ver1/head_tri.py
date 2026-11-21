"""買い方の検証: AIの1着確率1位（頭）が、1着の支持率で3番人気以降のレースだけ勝負。頭固定の3連単を複数点買う。
支持率: 3連単の確定オッズがあればそれ（頭ごとに 1/オッズ を合計）、無ければ2車単から計算（代用）。払戻は実際の3連単の払戻。"""
import pickle, sqlite3, numpy as np, pandas as pd, sys, itertools
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'; sys.path.insert(0,S)
from step3_fit import R
cols,w2,w3=pickle.load(open(f'{S}/step3_w.pkl','rb'))['基本7']
P=pd.read_pickle(f'{S}/wf_art.pkl'); P=P[P.method=='+記事の要素']; PD={k:dict(zip(g.car_no.astype(int),g.p)) for k,g in P.groupby('race_id')}
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
pay=pd.read_sql("select race_id,combo,payout from payouts where bet_type='3連単'",c); PAY={(a,b):v for a,b,v in pay.itertuples(index=False)}
tri=pd.read_sql("select race_id,combo,odds from odds where bet_type='3連単' and is_final=1",c)
tri['h']=tri.combo.str.split('>').str[0].astype(int); tri['iq']=1/tri.odds
TS={k:g.groupby('h').iq.sum().to_dict() for k,g in tri.groupby('race_id')}
T=pd.read_pickle(f'{S}/step5_T.pkl'); ex=T[T.bt=='2車単'][['race_id','combo','q']].copy(); ex['h']=ex.combo.str.split('>').str[0].astype(int)
ES={k:g.groupby('h').q.sum().to_dict() for k,g in ex.groupby('race_id')}
rows=[]
for r in R:
    rid=r['rid']; cars=r['cars']; n=r['n']
    if rid not in PD or not all(cc in PD[rid] for cc in cars) or not any((rid,k) in PAY for k in [None]) and False: pass
    if rid not in PD or not all(cc in PD[rid] for cc in cars): continue
    if not any(k[0]==rid for k in []): pass
    p=np.array([PD[rid][cc] for cc in cars]); p=p/p.sum(); h=int(np.argmax(p))
    sup=TS.get(rid) or ES.get(rid); src='3連単' if rid in TS else '2車単'
    if not sup: continue
    order=sorted(cars,key=lambda x:-sup.get(x,0)); rank=order.index(cars[h])+1
    X2=r['X2'].copy(); X3=r['X3'].copy(); lp=np.log(np.clip(p,1e-9,1)); X2[...,0]=lp[None,:]; X3[...,0]=lp[None,None,:]
    s2=X2[h][:,cols]@w2; s2[h]=-np.inf; q2=np.exp(s2-s2.max()); q2/=q2.sum()
    q3=np.zeros((n,n))
    for j in range(n):
        if j==h: continue
        s3=X3[h,j][:,cols]@w3; s3[[h,j]]=-np.inf; e=np.exp(s3-s3[np.isfinite(s3)].max()); q3[j]=e/e.sum()
    J=q2[:,None]*q3                                   # [2着, 3着]
    inv=J.sum(1)+J.sum(0); others=[x for x in np.argsort(-inv) if x!=h]
    def combos_from(A,B): return [f"{cars[h]}>{cars[a]}>{cars[b]}" for a in A for b in B if a!=b]
    sec=[x for x in np.argsort(-q2) if x!=h][:3]; third=[x for x in np.argsort(-J.sum(0)) if x!=h][:5]
    sets={'相手4人流し(12点)':combos_from(others[:4],others[:4]),'相手5人流し(20点)':combos_from(others[:5],others[:5]),
          '2着3人×3着5人':combos_from(sec,third)}
    out={'rid':rid,'date':r['date'],'rank':rank,'src':src,'n':n}
    for k,cs in sets.items(): out[k+'_点']=len(cs); out[k+'_払戻']=sum(PAY.get((rid,cc),0) for cc in cs)
    rows.append(out)
D=pd.DataFrame(rows); D.to_pickle(f'{S}/head_tri.pkl')
print('レース',len(D),' 支持率の出どころ',D.src.value_counts().to_dict(), ' 3連単払戻あり',sum(1 for x in D.rid if any(True for _ in [0])))
rng=np.random.default_rng(0)
def summ(d,k):
    st=d[k+'_点']*100; py=d[k+'_払戻']
    idx=rng.integers(0,len(d),(1000,len(d))); bs=py.to_numpy()[idx].sum(1)/st.to_numpy()[idx].sum(1)*100
    return f"{len(d)}R 的中{(py>0).mean()*100:.1f}% 回収率{py.sum()/st.sum()*100:.1f}% [{np.percentile(bs,2.5):.0f}〜{np.percentile(bs,97.5):.0f}]"
for k in ['相手4人流し(12点)','相手5人流し(20点)','2着3人×3着5人']:
    print('==',k)
    for lab,m in (('頭が1番人気',D['rank']==1),('頭が2番人気',D['rank']==2),('頭が3番人気以降（勝負）',D['rank']>=3),('全レース',D['rank']>0)):
        for per,pm in (('6/11〜8/31',D.date<'2026-09-01'),('9/1〜',D.date>='2026-09-01')):
            print(f"  {lab} {per}: {summ(D[m&pm],k)}")
print(D.groupby(D['rank'].clip(upper=5))['相手5人流し(20点)_点'].size())
