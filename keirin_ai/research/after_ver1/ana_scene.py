"""穴党の新しい予想方法（ver2候補）: 展開で穴が出る場面の3連単（20〜100倍）が、人気より安く見られているか。
場面の定義は結果を見る前に固定（2026-11-22）。人気の確率 q = 1/オッズ をレース内で合計1にしたもの（3連単の確定オッズ）。
S1 格上番手の不発: 番手の競走得点が先頭より3点以上高いライン → そのライン以外の選手が1着の目
S2 スジ違い: 一番強い先頭（得点1位の先頭）が1着、2着はそのラインの番手以外の目（番手が先頭より5点以上低いライン）
S3 点数差の小さい自力の逆転: 先頭どうしの得点差が2点未満で、低い方の先頭が1着になる目
"""
import sqlite3, numpy as np, pandas as pd
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
od=pd.read_sql("select race_id,combo,odds from odds where bet_type='3連単' and is_final=1",c)
pay=pd.read_sql("select race_id,combo,payout from payouts where bet_type='3連単'",c); WIN=dict(zip(pay.race_id,pay.combo))
E=pd.read_sql("select e.race_id,e.car_no,e.score,e.line_id,e.line_pos,e.line_size,e.scratched,r.date,r.race_class from entries e join races r using(race_id)",c)
E=E[E.race_id.isin(set(od.race_id))]
scr=set(E[E.scratched==1].race_id)
od=od[~od.race_id.isin(scr)&od.race_id.isin(set(WIN))]
od['q']=(1/od.odds)/od.groupby('race_id').odds.transform(lambda s:(1/s).sum())
od[['a','b','c']]=od.combo.str.split('>',expand=True).astype(int)
od['hit']=[WIN.get(r)==cb for r,cb in zip(od.race_id,od.combo)]
rows=[]
for rid,g in E.groupby('race_id'):
    if rid not in WIN: continue
    g=g.set_index('car_no'); g['score']=pd.to_numeric(g.score,errors='coerce')
    multi=g[g.line_size.fillna(1)>=2]
    leaders=multi[multi.line_pos==1]; bans=multi[multi.line_pos==2]
    lines={}
    for lid,m in multi.groupby('line_id'):
        m=m.sort_values('line_pos'); lines[lid]=[int(x) for x in m.index]
    s1=set(); s2=None; s3=set()
    for lid,cars in lines.items():
        if len(cars)>=2 and g.loc[cars[1],'score']-g.loc[cars[0],'score']>=3: s1|=set(cars)   # 格上番手のライン
    if len(leaders):
        top=leaders.score.idxmax(); lid=leaders.loc[top,'line_id']; cars=lines.get(lid,[])
        if len(cars)>=2 and g.loc[cars[0],'score']-g.loc[cars[1],'score']>=5: s2=(int(top),int(cars[1]))
        ls=leaders.score.sort_values(ascending=False)
        if len(ls)>=2 and ls.iloc[0]-ls.iloc[1]<2: s3.add(int(ls.index[1]))
    rows.append((rid,g.date.iloc[0],str(g.race_class.iloc[0]),s1,s2,s3))
R={r[0]:r for r in rows}
o=od[od.race_id.isin(R)].copy()
o['date']=o.race_id.map(lambda r:R[r][1]); o['cls']=o.race_id.map(lambda r:R[r][2])
o['S1']=[bool(R[r][3]) and a not in R[r][3] for r,a in zip(o.race_id,o.a)]
o['S2']=[R[r][4] is not None and a==R[r][4][0] and b!=R[r][4][1] for r,a,b in zip(o.race_id,o.a,o.b)]
o['S3']=[a in R[r][5] for r,a in zip(o.race_id,o.a)]
o['中穴']=o.odds.between(20,100)
rng=np.random.default_rng(0)
def summ(d):
    if d.race_id.nunique()<50: return f"{d.race_id.nunique()}R（少ない）"
    races=d.race_id.unique(); grp={r:i for i,r in enumerate(races)}; gi=d.race_id.map(grp).to_numpy()
    hit=np.bincount(gi,weights=d.hit.to_numpy(float),minlength=len(races)); q=np.bincount(gi,weights=d.q.to_numpy(),minlength=len(races))
    py=np.bincount(gi,weights=(d.hit*d.odds*100).to_numpy(),minlength=len(races)); st=np.bincount(gi,minlength=len(races))*100.
    b1=[];b2=[]
    for _ in range(500):
        i=rng.integers(0,len(races),len(races)); b1.append(hit[i].sum()/q[i].sum()); b2.append(py[i].sum()/st[i].sum()*100)
    return (f"{len(races)}R {len(d)}点 的中{int(hit.sum())}回 / 見込み{q.sum():.1f}回 = {hit.sum()/q.sum():.2f} [{np.percentile(b1,2.5):.2f}〜{np.percentile(b1,97.5):.2f}] | "
            f"全部買った回収率 {py.sum()/st.sum()*100:.0f}% [{np.percentile(b2,2.5):.0f}〜{np.percentile(b2,97.5):.0f}]")
print('対象',o.race_id.nunique(),'レース',o.date.min(),'〜',o.date.max())
print('参考: 中穴(20〜100倍)の目を全部:',summ(o[o.中穴]))
for s,lab in (('S1','格上番手の不発（そのライン以外が1着）'),('S2','スジ違い（強い先頭が1着・番手は2着に来ない）'),('S3','点数差2点未満の自力の逆転（低い方の先頭が1着）')):
    d=o[o[s]&o.中穴]
    print(f"{s} {lab}\n   全体: {summ(d)}\n   S級: {summ(d[d.cls.str.contains('S')])}\n   A級ほか: {summ(d[~d.cls.str.contains('S')])}")
o.to_pickle(f'{S}/ana_scene.pkl')
