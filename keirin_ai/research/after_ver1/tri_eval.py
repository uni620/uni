"""3連単: オッズ無しで買える1レースN点の買い方の回収率（実際の払戻）。期間 6/11〜11/20、AIの係数は 6/11〜8/31 で決めたもの。"""
import pickle, numpy as np, pandas as pd, sys
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'; sys.path.insert(0,S)
from step3_fit import R
cols,w2,w3=pickle.load(open(f'{S}/step3_w.pkl','rb'))['基本7']
TP=pd.read_pickle(f'{S}/tri_pay.pkl'); PAY={(r.race_id,r.combo):r.payout for r in TP.itertuples()}
F=pd.read_pickle(f'{S}/feats_0601.pkl')[['race_id','car_no','score','line_id','line_pos','line_size']]
F['score']=pd.to_numeric(F.score,errors='coerce'); FG={k:g for k,g in F.groupby('race_id')}
def ordered(r,w2,w3,cols):
    n=r['n']; p=r['p']
    s2=r['X2'][...,cols]@w2; s2[np.arange(n),np.arange(n)]=-np.inf
    q2=np.exp(s2-s2.max(1,keepdims=True)); q2/=q2.sum(1,keepdims=True)
    s3=r['X3'][...,cols]@w3; I=np.arange(n)
    s3[I[:,None],:,I[:,None]]=-np.inf; s3[:,I,I]=-np.inf
    s3[I,I,:]=-np.inf
    mx=np.max(np.where(np.isfinite(s3),s3,-1e300),axis=2,keepdims=True); e3=np.nan_to_num(np.exp(s3-mx)); q3=e3/np.maximum(e3.sum(2,keepdims=True),1e-300)
    P=p[:,None,None]*q2[:,:,None]*q3; P[I,I,:]=0; return P
H=[1.0,0,0,0,0,0,0]
rows=[]
for r in R:
    if r['rid'] not in FG: continue
    cars=r['cars']; n=r['n']
    out={'date':r['date'],'rid':r['rid']}
    for lab,(a,b) in (('ライン連動',(w2,w3)),('ハービル',(np.array(H),np.array(H)))):
        P=ordered(r,a,b,cols); flat=np.argsort(P.ravel())[::-1]
        picks=[np.unravel_index(k,P.shape) for k in flat[:12]]
        picks=[f"{cars[i]}>{cars[j]}>{cars[k]}" for i,j,k in picks]
        for N in (1,3,6,12):
            out[f'{lab}{N}点']=sum(PAY.get((r['rid'],c),0) for c in picks[:N])/N
    # 記事式: 競走得点の平均が一番高いラインのスジ（先頭>番手>3番手、2人ラインなら3着はAIの確率が一番高い残り）
    g=FG[r['rid']].set_index('car_no').loc[cars]
    lk=g.line_id.fillna(-pd.Series(cars,index=cars)).astype(int)
    lm=g.groupby(lk).score.mean(); best=lm.idxmax(); L=g[lk==best].sort_values('line_pos')
    p=dict(zip(cars,r['p']))
    if len(L)>=2:
        a,b=int(L.index[0]),int(L.index[1])
        c=int(L.index[2]) if len(L)>=3 else max((x for x in cars if x not in (a,b)),key=lambda x:p[x])
        out['最強ラインのスジ1点']=PAY.get((r['rid'],f"{a}>{b}>{c}"),0)
        out['最強ライン 番手まくり1点(番手>先頭>3人目)']=PAY.get((r['rid'],f"{b}>{a}>{c}"),0)
    rows.append(out)
D=pd.DataFrame(rows); rng=np.random.default_rng(0)
def ci(v):
    v=v.dropna().to_numpy(); idx=rng.integers(0,len(v),(2000,len(v))); return np.percentile(v[idx].mean(1),[2.5,97.5])
res=[]
for c in [x for x in D.columns if x not in ('date','rid')]:
    for per,m in (('全期間',D.date>='0'),('9/1〜',D.date>='2026-09-01')):
        v=D.loc[m,c].dropna(); lo,hi=ci(v)
        res.append(dict(買い方=c,期間=per,レース=len(v),的中率=round((v>0).mean()*100 if '1点' in c else np.nan,1),回収率=round(v.mean(),1),範囲=f"{lo:.0f}〜{hi:.0f}"))
pd.set_option('display.width',250); print(pd.DataFrame(res).to_string(index=False))
