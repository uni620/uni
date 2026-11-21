"""Mr.Tさん型の買い方を過去レースで試す（3連単・頭1人）。
頭: 一番先行力のある選手（逃げの決まり手の割合が最も高い先頭）がいるラインの「番手」（H1）か、そのラインの先頭と番手のうちAIの1着確率が高い方（H2）。
2着: 頭と同じラインの選手（最大2人）＋別ラインでAIが一番高い選手。3着: 頭以外でAIの上位5人（または6人）。
払戻は実際の3連単の配当。期間 6/11〜11/20。条件は全部、結果を見る前に決めたもの。"""
import sqlite3, numpy as np, pandas as pd
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
PAY={(a,b):v for a,b,v in pd.read_sql("select race_id,combo,payout from payouts where bet_type='3連単'",c).itertuples(index=False)}
F=pd.read_pickle(f'{S}/feats_0601.pkl'); F=F[F.date>='2026-06-11']
P=pd.read_pickle(f'{S}/wf_art.pkl'); P=P[P.method=='+記事の要素'][['race_id','car_no','p']]
F=F.merge(P,on=['race_id','car_no'])
T=pd.read_pickle(f'{S}/step5_T.pkl'); ex=T[T.bt=='2車単'][['race_id','combo','q']].copy(); ex['h']=ex.combo.str.split('>').str[0].astype(int)
WQ=ex.groupby(['race_id','h']).q.sum()
scr=set(pd.read_sql("select distinct race_id from entries where scratched=1",c).race_id)
rows=[]
for rid,g in F.groupby('race_id'):
    if rid in scr or not any(k[0]==rid for k in [] ) and (rid,'x') in PAY: pass
    g=g.copy(); g['car']=g.car_no.astype(int); g=g.set_index('car')
    if g.finish.isna().all(): continue
    p=g.p/g.p.sum()
    kn=g.kimarite_n.replace(0,np.nan); g['nige']=(g.km_nige/kn).fillna(0)
    multi=g[(g.line_size.fillna(1)>=2)]
    leaders=multi[multi.line_pos==1]
    if leaders.empty: continue
    L=leaders.nige.idxmax(); lid=g.loc[L,'line_id']
    mates=g[(g.line_id==lid)].sort_values('line_pos')
    if len(mates)<2: continue
    ban=int(mates.index[1])
    heads={'H1 番手':ban,'H2 先頭か番手(AIが上)':ban if p[ban]>=p[L] else int(L)}
    try: rank={h:int((-WQ.loc[rid]).rank().get(h)) for h in set(heads.values())}
    except KeyError: continue
    for hn,h in heads.items():
        same=[int(x) for x in g[(g.line_id==g.loc[h,'line_id'])&(g.index!=h)].sort_values('line_pos').index[:2]]
        oth=p.drop([h]+same).sort_values(ascending=False)
        sec=same+[int(oth.index[0])]
        for n3 in (5,6):
            third=[int(x) for x in p.drop(h).sort_values(ascending=False).index[:n3]]
            cs=[f"{h}>{a}>{b}" for a in sec for b in third if a!=b]
            rows.append(dict(rid=rid,date=g.date.iloc[0],head=hn,n3=n3,pts=len(cs),pay=sum(PAY.get((rid,x),0) for x in cs),
                             cls=str(g.race_class.iloc[0]),grade=str(g.grade.iloc[0]),rank=rank[h],head_is_ban=h==ban))
D=pd.DataFrame(rows); D.to_pickle(f'{S}/mrt_rule.pkl')
rng=np.random.default_rng(0)
def summ(d):
    if len(d)<30: return f"{len(d)}R（少なすぎ）"
    st=d.pts.to_numpy()*100; py=d.pay.to_numpy(); idx=rng.integers(0,len(d),(1000,len(d))); bs=py[idx].sum(1)/st[idx].sum(1)*100
    return f"{len(d)}R 平均{d.pts.mean():.0f}点 的中{(d.pay>0).mean()*100:.1f}% 回収率{py.sum()/st.sum()*100:.1f}% [{np.percentile(bs,2.5):.0f}〜{np.percentile(bs,97.5):.0f}]"
SEL={'全レース':lambda d:d.rid!='','S級だけ':lambda d:d.cls.str.contains('S'),'G戦だけ(G1〜G3)':lambda d:d.grade.str.startswith('G'),
     '頭が1番人気でない':lambda d:d['rank']>=2,'S級×頭が1番人気でない':lambda d:d.cls.str.contains('S')&(d['rank']>=2)}
for hn in D['head'].unique():
    for n3 in (5,6):
        print(f"== {hn} / 3着{n3}人")
        for sn,f in SEL.items():
            d=D[(D['head']==hn)&(D.n3==n3)]; d=d[f(d)]
            print(f"  {sn}: 6/11〜8/31 {summ(d[d.date<'2026-09-01'])} | 9/1〜 {summ(d[d.date>='2026-09-01'])}")
