"""番手の選手で、1着の人気が4〜6番目の選手を頭にして勝負するシミュレーション（6/11〜11/20、実際の払戻）。
人気は2車単の確定オッズから計算した1着支持率の順位。1レースに該当が複数いればそれぞれ買う。"""
import sqlite3, numpy as np, pandas as pd
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
pay=pd.read_sql("select race_id,bet_type,combo,payout from payouts where bet_type in ('2車単','3連単')",c)
PAY={(a,b,cc):v for a,b,cc,v in pay.itertuples(index=False)}
F=pd.read_pickle(f'{S}/feats_0601.pkl'); F=F[F.date>='2026-06-11'][['race_id','date','car_no','finish','line_id','line_pos','line_size','race_class']]
P=pd.read_pickle(f'{S}/wf_art.pkl'); P=P[P.method=='+記事の要素'][['race_id','car_no','p']]
T=pd.read_pickle(f'{S}/step5_T.pkl'); ex=T[T.bt=='2車単'][['race_id','combo','q']].copy(); ex['car_no']=ex.combo.str.split('>').str[0].astype(int)
W=ex.groupby(['race_id','car_no']).q.sum().reset_index()
D=F.merge(W,on=['race_id','car_no']).merge(P,on=['race_id','car_no'],how='left')
D['rank']=D.groupby('race_id').q.rank(ascending=False,method='first')
scr=set(pd.read_sql("select distinct race_id from entries where scratched=1",c).race_id)
rows=[]
for rid,g in D.groupby('race_id'):
    if rid in scr: continue
    g=g.set_index(g.car_no.astype(int))
    for h,r in g.iterrows():
        if not(r.line_size>=2 and r.line_pos==2): continue
        mates=g[(g.line_id==r.line_id)]
        lead=int(mates[mates.line_pos==1].index[0]) if (mates.line_pos==1).any() else None
        if lead is None: continue
        others=g.drop([h]).sort_values('p',ascending=False)
        top5=[int(x) for x in others.index[:5]]
        sec=[lead]+[int(x) for x in others.drop(lead).index[:2]]
        bets={'2車単 番手→先頭（1点）':[('2車単',f"{h}>{lead}")],
              '2車単 番手→全員（流し）':[('2車単',f"{h}>{x}") for x in g.index if x!=h],
              '3連単 番手→先頭→AI上位5人（5点前後）':[('3連単',f"{h}>{lead}>{x}") for x in top5 if x!=lead],
              '3連単 番手→先頭+AI2人→AI上位5人（約12点）':[('3連単',f"{h}>{a}>{b}") for a in sec for b in top5 if a!=b]}
        won=float(pd.to_numeric(r.finish,errors='coerce')==1)
        for k,bs in bets.items():
            rows.append(dict(rid=rid,date=r.date,rank=int(r['rank']),q=r.q,won=won,way=k,pts=len(bs),pay=sum(PAY.get((rid,bt,cb),0) for bt,cb in bs),cls=str(r.race_class)))
X=pd.DataFrame(rows); X.to_pickle(f'{S}/banete46.pkl')
rng=np.random.default_rng(0)
def summ(d):
    if len(d)<30: return f"{len(d)}件（少なすぎ）"
    st=d.pts.to_numpy()*100.; py=d.pay.to_numpy(); idx=rng.integers(0,len(d),(1000,len(d))); bs=py[idx].sum(1)/st[idx].sum(1)*100
    return f"{len(d)}件 的中{(d.pay>0).mean()*100:.1f}% 回収率{py.sum()/st.sum()*100:.1f}% [{np.percentile(bs,2.5):.0f}〜{np.percentile(bs,97.5):.0f}]"
one=X[X.way=='2車単 番手→先頭（1点）']
for lab,m in (('番手・1〜3番人気',one['rank']<=3),('番手・4〜6番人気',one['rank'].between(4,6)),('番手・7番人気以下',one['rank']>=7)):
    d=one[m]; print(f"{lab}: {len(d)}人 1着{int(d.won.sum())}回 / 人気の見込み{d.q.sum():.0f}回 = {d.won.sum()/d.q.sum():.3f}")
for way in X.way.unique():
    print('==',way)
    d=X[(X.way==way)&X['rank'].between(4,6)]
    print('  全期間  ',summ(d)); print('  6/11〜8/31',summ(d[d.date<'2026-09-01'])); print('  9/1〜   ',summ(d[d.date>='2026-09-01']))
    print('  S級だけ ',summ(d[d.cls.str.contains('S')]),' / A級ほか',summ(d[~d.cls.str.contains('S')]))
