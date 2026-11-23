"""オッズだけの確率（4券種）→ レースごとに券種をまたいで期待値の高い目だけ買う。前半学習→後半確認。"""
import pandas as pd, numpy as np, sys
exec(open('cross2.py').read().split("rng=np.random")[0].split("CUT=")[0])   # LogisticRegression 定義だけ借りる
CUT=sys.argv[1] if len(sys.argv)>1 else '2026-10-15'
MOD={}
d=pd.read_pickle('odds_val.pkl'); d=d[d.method=='LightGBM'][['race_id','date','cls','nr','bt','combo','odds','ret','q0']].copy()
d['q']=d.q0/d.groupby(['race_id','bt']).q0.transform('sum')
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
    y=(x.ret>0).astype(int).values; t=(x.date<CUT).values
    lr=LogisticRegression().fit(F[t],y[t]); x['pm']=lr.predict_proba(F)[:,1]; x['ev']=x.pm*x.odds
    MOD[bt]={'cols':cols,'mu':lr.mu,'sd':lr.sd,'w':lr.w}
    te=x[~t]; print(bt,'後半 当たり具合', round(te.ret.gt(0).sum()/te.pm.sum(),3), flush=True)
    res.append(x[~t][['race_id','date','bt','combo','odds','ret','pm','ev']])
pd.to_pickle(MOD,f'cross3_model_{CUT}.pkl')
if CUT<'2026-11-20': T=pd.concat(res); T.to_pickle('cross3_test.pkl')
