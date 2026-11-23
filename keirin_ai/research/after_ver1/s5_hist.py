"""玉砕型の先頭を「過去の走り方」で判定（コメントは使わない）。定義は結果を見る前に固定（2026-11-22）。
先頭で走った過去レースのうち: 先行率 = B（バック先頭）を取った割合、垂れ率 = Bを取ったときに4着以下だった割合。
玉砕型: 先頭で5回以上走り、先行率0.4以上、Bを取った回数3回以上で垂れ率0.5以上、かつ今回の1着人気が3番目以下。
対象: その番手で1着人気4番目以下。"""
import sqlite3, numpy as np, pandas as pd
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite',timeout=60)
E=pd.read_sql("""select e.race_id,e.car_no,e.rider_id,e.score,e.finish,e.line_id,e.line_pos,e.line_size,e.scratched,r.date,r.race_class,i.sb
                 from entries e join races r using(race_id) left join result_info i using(race_id,car_no) where r.has_result=1""",c)
E['finish']=pd.to_numeric(E.finish,errors='coerce')
L=E[(E.line_size>=2)&(E.line_pos==1)&(E.scratched!=1)&E.finish.notna()].copy()
L['B']=L.sb.astype(str).str.contains('B').astype(float); L['tare']=((L.B==1)&(L.finish>=4)).astype(float); L['n']=1.0
g=L.groupby(['rider_id','date'])[['n','B','tare']].sum(); cum=g.groupby(level=0).cumsum()-g
H=cum.reset_index().rename(columns={'n':'h_n','B':'h_B','tare':'h_tare'})
E=E.merge(H,on=['rider_id','date'],how='left')
# 人気（2車単の確定オッズ）
od=pd.read_sql("select race_id,combo,odds from odds where is_final=1 and bet_type='2車単'",c)
od=od.drop_duplicates(['race_id','combo']); od['h']=od.combo.str.split('>').str[0].astype(int); od['iq']=1/od.odds
W=od.groupby(['race_id','h']).iq.sum().reset_index().rename(columns={'h':'car_no'}); W['q']=W.iq/W.groupby('race_id').iq.transform('sum')
E=E.merge(W[['race_id','car_no','q']],on=['race_id','car_no'],how='left')
E['rank']=E.groupby('race_id').q.rank(ascending=False,method='first')
E['won']=(E.finish==1).astype(float)
scr=set(E[E.scratched==1].race_id); E=E[~E.race_id.isin(scr)]
ld=E[(E.line_size>=2)&(E.line_pos==1)].copy()
ld['B率']=ld.h_B/ld.h_n; ld['垂れ率']=ld.h_tare/ld.h_B.replace(0,np.nan)
ld['玉砕型']=(ld.h_n>=5)&(ld['B率']>=0.4)&(ld.h_B>=3)&(ld['垂れ率']>=0.5)&(ld['rank']>=3)
bn=E[(E.line_size>=2)&(E.line_pos==2)].merge(ld[['race_id','line_id','玉砕型']],on=['race_id','line_id'])
bn=bn[bn.q.notna()]
rng=np.random.default_rng(0)
def ratio(d):
    if len(d)<30: return f"{len(d)}人（少ない）"
    a=d[['won','q']].to_numpy(); b=[]
    for _ in range(500):
        i=rng.integers(0,len(a),len(a)); b.append(a[i,0].sum()/a[i,1].sum())
    return f"{len(d)}人 1着{int(a[:,0].sum())} / 見込み{a[:,1].sum():.1f} = {a[:,0].sum()/a[:,1].sum():.2f} [{np.percentile(b,2.5):.2f}〜{np.percentile(b,97.5):.2f}]"
print('オッズのある期間:',bn.date.min(),'〜',bn.date.max())
x=bn[bn['rank']>=4]
print('人気薄の番手・先頭が玉砕型    ',ratio(x[x.玉砕型]))
print('人気薄の番手・先頭が玉砕型でない',ratio(x[~x.玉砕型]))
for per,m in (('〜6/10',x.date<'2026-06-11'),('6/11〜8/31',(x.date>='2026-06-11')&(x.date<'2026-09-01')),('9/1〜',x.date>='2026-09-01')):
    print('   ',per,'玉砕型:',ratio(x[x.玉砕型&m]))
bn.to_pickle(f'{S}/s5_bn.pkl'); ld.to_pickle(f'{S}/s5_ld.pkl')
