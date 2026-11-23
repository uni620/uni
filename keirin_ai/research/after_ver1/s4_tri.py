"""S4 の3連単シミュレーション: 頭＝人気薄の番手（玉砕型の先頭の番手・1着人気4番目以下）、その先頭は2・3着から除く。
買い目（結果を見る前に固定）:
 B1 番手→（先頭以外の全員）→（先頭以外の全員）
 B2 番手→AI上位3人→AI上位5人（どちらも先頭を除く）
 B3 番手→AI上位4人の中で2・3着（先頭を除く、12点）
 B4 番手→別ラインの先頭たち→先頭以外の全員
期間 6/11〜11/20、払戻は実際の3連単の配当。"""
import sqlite3, numpy as np, pandas as pd
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
exec(open(f'{S}/s4.py').read().split("rng=np.random")[0])     # ban, lead, D を作る部分だけ使う
P=pd.read_pickle(f'{S}/wf_art.pkl'); P=P[P.method=='+記事の要素']; PD={k:dict(zip(g.car_no.astype(int),g.p)) for k,g in P.groupby('race_id')}
pay=pd.read_sql("select race_id,combo,payout from payouts where bet_type='3連単'",c); WIN=dict(zip(pay.race_id,pay.combo)); PAYV=dict(zip(pay.race_id,pay.payout))
G={k:g for k,g in D.groupby('race_id')}
rows=[]
for typ,mask in (('コメント型',ban.薄&ban.g_cmt),('成績型',ban.薄&ban.g_stat)):
    for r in ban[mask].itertuples():
        rid=r.race_id
        if rid not in WIN or rid not in PD: continue
        g=G[rid]; cars=[int(x) for x in g.car_no]; h=int(r.car_no)
        L=int(g[(g.line_id==r.line_id)&(g.line_pos==1)].car_no.iloc[0])
        others=[x for x in cars if x not in (h,L)]
        p=PD[rid]; ai=sorted(others,key=lambda x:-p.get(x,0))
        oleads=[int(x) for x in g[(g.line_size>=2)&(g.line_pos==1)&(g.line_id!=r.line_id)].car_no]
        sets={'B1 先頭以外の全員に流す':[(a,b) for a in others for b in others if a!=b],
              'B2 2着AI3人×3着AI5人':[(a,b) for a in ai[:3] for b in ai[:5] if a!=b],
              'B3 AI上位4人で2・3着（12点）':[(a,b) for a in ai[:4] for b in ai[:4] if a!=b],
              'B4 2着は別ラインの先頭':[(a,b) for a in oleads for b in others if a!=b]}
        w=WIN[rid]
        for k,cs in sets.items():
            hit=any(f"{h}>{a}>{b}"==w for a,b in cs)
            rows.append(dict(typ=typ,way=k,rid=rid,date=r.date,pts=len(cs),pay=PAYV[rid] if hit else 0,head_won=float(w.split('>')[0]==str(h)),leader_top3=str(L) in w.split('>')))
X=pd.DataFrame(rows); rng=np.random.default_rng(0)
def summ(d):
    if len(d)<20: return f"{len(d)}R（少ない）"
    st=d.pts.to_numpy()*100.; py=d.pay.to_numpy(); i=rng.integers(0,len(d),(1000,len(d))); b=py[i].sum(1)/st[i].sum(1)*100
    return f"{len(d)}R 平均{d.pts.mean():.0f}点 的中{(d.pay>0).sum()}回 回収率{py.sum()/st.sum()*100:.0f}% [{np.percentile(b,2.5):.0f}〜{np.percentile(b,97.5):.0f}]"
for typ in ('コメント型','成績型'):
    d0=X[(X.typ==typ)&(X.way.str.startswith('B1'))]
    print(f"== {typ}: 番手の1着 {int(d0.head_won.sum())}/{len(d0)}  玉砕型の先頭が3着以内 {d0.leader_top3.mean()*100:.0f}%")
    for way in X.way.unique():
        d=X[(X.typ==typ)&(X.way==way)]
        print(f"  {way}: 全期間 {summ(d)} | 6/11〜8/31 {summ(d[d.date<'2026-09-01'])} | 9/1〜 {summ(d[d.date>='2026-09-01'])}")
X.to_pickle(f'{S}/s4_tri.pkl')
