"""ver2候補（S4: コメント型の先頭×人気薄の番手×ライン強さ1〜2位×3連単12点）を別の期間（〜6/10）で試す。
オッズが無い期間なので、人気の代わりに競走得点の順位を使う（番手の得点順位4位以下＝人気薄の代わり）。
2着・3着の候補も AI の代わりに競走得点の上位4人（頭を除く、先頭も入れる）。比較のため 6/11〜も同じ代わりの条件で計算する。"""
import sqlite3, numpy as np, pandas as pd
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
E=pd.read_sql("select e.race_id,e.car_no,e.score,e.finish,e.line_id,e.line_pos,e.line_size,e.scratched,r.date,r.race_class from entries e join races r using(race_id) where r.has_result=1",c)
E=E[~E.race_class.astype(str).str.contains('ガールズ')]
M=pd.read_sql('select race_id,car_no,comment from entry_info',c)
E=E.merge(M,on=['race_id','car_no'],how='left')
pay=pd.read_sql("select race_id,combo,payout from payouts where bet_type='3連単'",c); WIN=dict(zip(pay.race_id,pay.combo)); PV=dict(zip(pay.race_id,pay.payout))
scr=set(E[E.scratched==1].race_id)
rows=[]
for rid,g in E.groupby('race_id'):
    if rid in scr or rid not in WIN: continue
    g=g.copy(); g['score']=pd.to_numeric(g.score,errors='coerce'); g['srank']=g.score.rank(ascending=False,method='first')
    g['lk']=g.line_id.fillna(-g.car_no).astype(int); lm=g.groupby('lk').score.mean().rank(ascending=False,method='min')
    for _,b in g[(g.line_size>=2)&(g.line_pos==2)].iterrows():
        L=g[(g.line_id==b.line_id)&(g.line_pos==1)]
        if L.empty: continue
        L=L.iloc[0]
        if not str(L.comment).__contains__('前々') and not any(k in str(L.comment) for k in ('先行','逃げ','駆け','積極','主導権')): continue
        if b.srank<4: continue
        lr=lm[int(b.line_id)]; h=int(b.car_no)
        cand=[int(x) for x in g[g.car_no!=h].sort_values('score',ascending=False).car_no[:4]]
        cs=[f"{h}>{a}>{x}" for a in cand for x in cand if a!=x]
        w=WIN[rid]
        rows.append(dict(rid=rid,date=g.date.iloc[0],lr=lr,pts=len(cs),pay=PV[rid] if w in cs else 0,won=w.split('>')[0]==str(h)))
X=pd.DataFrame(rows); rng=np.random.default_rng(0)
def summ(d):
    if len(d)<20: return f"{len(d)}R（少ない）"
    st=d.pts.to_numpy()*100.; py=d.pay.to_numpy(); i=rng.integers(0,len(d),(1000,len(d))); b=py[i].sum(1)/st[i].sum(1)*100
    return f"{len(d)}R 番手1着{int(d.won.sum())} 的中{(d.pay>0).sum()} 回収率{py.sum()/st.sum()*100:.0f}% [{np.percentile(b,2.5):.0f}〜{np.percentile(b,97.5):.0f}]"
print('期間',X.date.min(),'〜',X.date.max())
for per,m in (('〜6/10（別の期間）',X.date<'2026-06-11'),('6/11〜11/20（同じ代わりの条件）',X.date>='2026-06-11')):
    d=X[m]; print(per); print('   ライン強さ1〜2位:',summ(d[d.lr<=2])); print('   ライン強さ3位以下:',summ(d[d.lr>=3])); print('   全部:',summ(d))
