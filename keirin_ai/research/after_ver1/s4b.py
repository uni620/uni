"""S4 の続き（探索）: 先頭を外さない買い方・ラインの強さ・先頭の過去の玉砕回数で絞る。6/11〜11/20、実際の3連単の配当。"""
import sqlite3, numpy as np, pandas as pd
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
exec(open(f'{S}/s4.py').read().split("rng=np.random")[0])
# --- 先頭の過去の玉砕回数（そのレースより前の日だけ）: 先頭で走り、自分は4着以下・番手は2着以内
E2=pd.read_sql("select e.race_id,e.rider_id,e.finish,e.line_id,e.line_pos,e.line_size,e.scratched,r.date from entries e join races r using(race_id)",c)
E2['finish']=pd.to_numeric(E2.finish,errors='coerce'); E2=E2[(E2.scratched!=1)&(E2.line_size>=2)]
ld=E2[E2.line_pos==1][['race_id','line_id','rider_id','finish','date']]; bn=E2[E2.line_pos==2][['race_id','line_id','finish']].rename(columns={'finish':'bf'})
H=ld.merge(bn,on=['race_id','line_id']).dropna(subset=['finish','bf'])
H['gyoku']=((H.finish>=4)&(H.bf<=2)).astype(float); H['n']=1.0
g=H.groupby(['rider_id','date'])[['gyoku','n']].sum().groupby(level=0).cumsum()-H.groupby(['rider_id','date'])[['gyoku','n']].sum()
g=g.reset_index()
Fr=pd.read_pickle(f'{S}/feats_0601.pkl')[['race_id','car_no','rider_id']]
lead2=lead.merge(Fr,on=['race_id','car_no']).merge(g.rename(columns={'gyoku':'g_cnt','n':'lead_n'}),on=['rider_id','date'],how='left').fillna({'g_cnt':0,'lead_n':0})
# --- ラインの強さ: ラインの平均得点の順位（1=一番強い）
D['score']=pd.to_numeric(D.score,errors='coerce')
D['lkey']=D.race_id+'_'+D.line_id.fillna(-D.car_no).astype(int).astype(str)
lm=D.groupby(['race_id','lkey']).score.mean().reset_index(); lm['line_rank']=lm.groupby('race_id').score.rank(ascending=False,method='min')
D=D.merge(lm[['lkey','line_rank']],on='lkey',how='left')
b2=ban.merge(lead2[['race_id','line_id','g_cnt','lead_n']],on=['race_id','line_id'],how='left')
b2['lkey']=b2.race_id+'_'+b2.line_id.astype(int).astype(str); b2=b2.merge(lm[['lkey','line_rank']],on='lkey',how='left')
P=pd.read_pickle(f'{S}/wf_art.pkl'); P=P[P.method=='+記事の要素']; PD={k:dict(zip(gg.car_no.astype(int),gg.p)) for k,gg in P.groupby('race_id')}
pay=pd.read_sql("select race_id,combo,payout from payouts where bet_type='3連単'",c); WIN=dict(zip(pay.race_id,pay.combo)); PAYV=dict(zip(pay.race_id,pay.payout))
G={k:gg for k,gg in D.groupby('race_id')}
base=b2[b2.薄&(b2.g_cmt|b2.g_stat|(b2.g_cnt>=2))]
rows=[]
for r in base.itertuples():
    rid=r.race_id
    if rid not in WIN or rid not in PD: continue
    gg=G[rid]; cars=[int(x) for x in gg.car_no]; h=int(r.car_no)
    L=int(gg[(gg.line_id==r.line_id)&(gg.line_pos==1)].car_no.iloc[0])
    p=PD[rid]; others=[x for x in cars if x not in (h,L)]; ai=sorted(others,key=lambda x:-p.get(x,0)); ai_all=sorted([x for x in cars if x!=h],key=lambda x:-p.get(x,0))
    sets={'B2 2着AI3人×3着AI5人（先頭を外す）':[(a,b) for a in ai[:3] for b in ai[:5] if a!=b],
          'B3 AI上位4人で2・3着（先頭を外す）':[(a,b) for a in ai[:4] for b in ai[:4] if a!=b],
          'B5 2着=先頭+AI2人×3着AI5人（先頭も入れる）':[(a,b) for a in [L]+ai[:2] for b in ([L]+ai)[:5] if a!=b],
          'B6 番手→先頭→AI上位5人（スジ）':[(L,b) for b in ai[:5]],
          'B7 AI上位4人で2・3着（先頭も候補）':[(a,b) for a in ai_all[:4] for b in ai_all[:4] if a!=b]}
    w=WIN[rid]
    for k,cs in sets.items():
        hit=any(f"{h}>{a}>{b}"==w for a,b in cs)
        rows.append(dict(way=k,rid=rid,date=r.date,pts=len(cs),pay=PAYV[rid] if hit else 0,cmt=r.g_cmt,stat=r.g_stat,g_cnt=r.g_cnt,line_rank=r.line_rank,won=float(w.split('>')[0]==str(h))))
X=pd.DataFrame(rows); rng=np.random.default_rng(0)
def summ(d):
    if len(d)<20: return f"{len(d)}R（少ない）"
    st=d.pts.to_numpy()*100.; py=d.pay.to_numpy(); i=rng.integers(0,len(d),(1000,len(d))); b=py[i].sum(1)/st[i].sum(1)*100
    return f"{len(d)}R 的中{(d.pay>0).sum()} 回収率{py.sum()/st.sum()*100:.0f}% [{np.percentile(b,2.5):.0f}〜{np.percentile(b,97.5):.0f}] (前半{(py[d.date.to_numpy()<'2026-09-01'].sum()/max(st[d.date.to_numpy()<'2026-09-01'].sum(),1)*100):.0f}%/後半{(py[d.date.to_numpy()>='2026-09-01'].sum()/max(st[d.date.to_numpy()>='2026-09-01'].sum(),1)*100):.0f}%)"
F={'コメント型':lambda d:d.cmt,'コメント型×ライン強さ1〜2位':lambda d:d.cmt&(d.line_rank<=2),'コメント型×ライン強さ3位以下':lambda d:d.cmt&(d.line_rank>=3),
   '玉砕2回以上（コメント問わず）':lambda d:d.g_cnt>=2,'玉砕2回以上×ライン強さ1〜2位':lambda d:(d.g_cnt>=2)&(d.line_rank<=2),
   'コメント型×玉砕1回以上':lambda d:d.cmt&(d.g_cnt>=1),'玉砕3回以上':lambda d:d.g_cnt>=3}
X.to_pickle(f'{S}/s4b.pkl')
for fn,f in F.items():
    d0=X[X.way.str.startswith('B6')]; d0=d0[f(d0)]
    print(f"== {fn}（番手1着 {int(d0.won.sum())}/{len(d0)}）")
    for way in X.way.unique():
        d=X[X.way==way]; print('   ',way,summ(d[f(d)]))
