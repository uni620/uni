import pandas as pd, numpy as np
d=pd.read_pickle('odds_val.pkl'); pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
nw=d[(d.ret>0)&(d.bt!='ワイド')&(d.method=='LightGBM')]
print("ワイド除く 一致率", ((nw.odds*100-nw.ret).abs()<=nw.ret*0.02+10).mean().round(3))
rng=np.random.default_rng(0)
def ci(g):
    r=g.groupby('race_id').agg(s=('ret','size'),r=('ret','sum'))
    if len(r)<5: return pd.Series({'R':len(r),'点':int(r.s.sum()),'回収%':np.nan,'下':np.nan,'上':np.nan,'的中%':np.nan})
    s,v=r.s.values*100.,r.r.values.astype(float); idx=rng.integers(0,len(r),(400,len(r)))
    b=v[idx].sum(1)/s[idx].sum(1)*100
    return pd.Series({'R':len(r),'点':int(s.sum()/100),'回収%':round(v.sum()/s.sum()*100,1),'下':round(np.percentile(b,2.5)),'上':round(np.percentile(b,97.5)),'的中%':round((g.ret>0).mean()*100,1)})
top=d[d.rk_p==1].copy()
top['evb']=pd.cut(top.ev,[0,0.8,1.0,1.2,1.5,2,1e9],labels=['<0.8','0.8-1','1-1.2','1.2-1.5','1.5-2','2~'])
print("\n[C] AI本命1点：期待値(AI確率×確定オッズ)の帯ごと")
for bt in ['2車複','2車単','3連複','ワイド']:
    x=top[(top.bt==bt)&(top.method=='LightGBM+競走得点')]
    print('--',bt,'(LightGBM+競走得点)'); print(x.groupby('evb',observed=True).apply(ci).to_string())
print("\n[D] AI本命1点：全部 vs 期待値1.2以上だけ（目安オッズ以上）")
out=[]
for (m,bt),x in top.groupby(['method','bt']):
    a=ci(x); b=ci(x[x.ev>=1.2]); out.append([m,bt,a['回収%'],f"{a['下']:.0f}-{a['上']:.0f}",a.R,b['回収%'],f"{b['下']:.0f}-{b['上']:.0f}",b.R])
print(pd.DataFrame(out,columns=['方法','券種','全部','95%幅','R','EV≥1.2','95%幅','R']).to_string())
print("\n[E] 全組み合わせから EV≥t を全部買う（AI上位10位まで・オッズ50倍以下）")
out=[]
for (m,bt),x in d[(d.rk_p<=10)&(d.odds<=50)].groupby(['method','bt']):
    for th in [1.0,1.2,1.5,2.0]:
        a=ci(x[x.ev>=th]); out.append([m,bt,th,a.R,a['点'],a['回収%'],f"{a['下']:.0f}-{a['上']:.0f}",a['的中%']])
print(pd.DataFrame(out,columns=['方法','券種','EV≥','R','点','回収%','95%幅','的中%']).to_string())
