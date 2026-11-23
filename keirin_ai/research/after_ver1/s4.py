"""S4 人気薄の番手が勝つレース（玉砕覚悟の先頭がいる場合）。定義は結果を見る前に固定（2026-11-22）。
玉砕型の先頭: ラインの先頭（2人以上のライン）で、決まり手の半分以上が逃げ（決まり手5回以上）、かつ1着人気が3番目以下。
（別版）コメント型: 先頭のコメントに 前々/先行/逃げ/駆け/積極/主導権 がある。
対象: その先頭の番手で、1着人気が4番目以下。"""
import sqlite3, numpy as np, pandas as pd, re
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
F=pd.read_pickle(f'{S}/feats_0601.pkl'); F=F[F.date>='2026-06-11'][['race_id','date','car_no','finish','line_id','line_pos','line_size','km_nige','kimarite_n','race_class','score']]
M=pd.read_pickle(f'{S}/honshi_marks.pkl')[['race_id','car_no','comment']]
T=pd.read_pickle(f'{S}/step5_T.pkl'); ex=T[T.bt=='2車単'][['race_id','combo','q']].copy(); ex['car_no']=ex.combo.str.split('>').str[0].astype(int)
W=ex.groupby(['race_id','car_no']).q.sum().reset_index()
D=F.merge(W,on=['race_id','car_no']).merge(M,on=['race_id','car_no'],how='left')
D['rank']=D.groupby('race_id').q.rank(ascending=False,method='first')
D['won']=(pd.to_numeric(D.finish,errors='coerce')==1).astype(float)
D['nige']=D.km_nige/D.kimarite_n.replace(0,np.nan)
scr=set(pd.read_sql("select distinct race_id from entries where scratched=1",c).race_id)
D=D[~D.race_id.isin(scr)]
lead=D[(D.line_size>=2)&(D.line_pos==1)].copy()
lead['g_stat']=(lead.nige>=0.5)&(lead.kimarite_n>=5)&(lead['rank']>=3)
lead['g_cmt']=lead.comment.astype(str).str.contains('前々|先行|逃げ|駆け|積極|主導権')
ban=D[(D.line_size>=2)&(D.line_pos==2)].merge(lead[['race_id','line_id','g_stat','g_cmt','rank']].rename(columns={'rank':'lead_rank'}),on=['race_id','line_id'])
ban['薄']=ban['rank']>=4
rng=np.random.default_rng(0)
def ratio(g):
    if len(g)<50: return f"{len(g)}人（少ない）"
    a=g[['won','q']].to_numpy(); b=[]
    for _ in range(500):
        i=rng.integers(0,len(a),len(a)); b.append(a[i,0].sum()/a[i,1].sum())
    return f"{len(g)}人 1着{int(a[:,0].sum())}回 / 見込み{a[:,1].sum():.1f}回 = {a[:,0].sum()/a[:,1].sum():.2f} [{np.percentile(b,2.5):.2f}〜{np.percentile(b,97.5):.2f}]"
print('== 番手の1着（2車単の人気との比較、6/11〜11/20）')
print(' 人気薄の番手・全体            ',ratio(ban[ban.薄]))
print(' 人気薄の番手・玉砕型の先頭(成績)',ratio(ban[ban.薄&ban.g_stat]))
print(' 人気薄の番手・玉砕型の先頭(コメント)',ratio(ban[ban.薄&ban.g_cmt]))
print(' 人気薄の番手・先頭が玉砕型でない',ratio(ban[ban.薄&~ban.g_stat&~ban.g_cmt]))
for per,m in (('6/11〜8/31',ban.date<'2026-09-01'),('9/1〜',ban.date>='2026-09-01')):
    print(f'   {per} 玉砕型(成績):',ratio(ban[ban.薄&ban.g_stat&m]))
# 3連単（10/1〜）: その番手が頭の 20〜100倍 / 全部
O=pd.read_pickle(f'{S}/ana_scene.pkl')
tgt=ban[ban.薄&(ban.g_stat|ban.g_cmt)][['race_id','car_no','g_stat','g_cmt']]
o=O.merge(tgt.rename(columns={'car_no':'a'}),on=['race_id','a'])
def summ(d):
    if d.race_id.nunique()<30: return f"{d.race_id.nunique()}R（少ない）"
    races=d.race_id.unique(); gi=d.race_id.map({r:i for i,r in enumerate(races)}).to_numpy()
    hit=np.bincount(gi,weights=d.hit.to_numpy(float)); q=np.bincount(gi,weights=d.q.to_numpy()); py=np.bincount(gi,weights=(d.hit*d.odds*100).to_numpy()); st=np.bincount(gi)*100.
    b=[]
    for _ in range(500):
        i=rng.integers(0,len(races),len(races)); b.append(py[i].sum()/st[i].sum()*100)
    return f"{len(races)}R {len(d)}点 的中{int(hit.sum())}回/見込み{q.sum():.1f}回={hit.sum()/q.sum():.2f} | 回収率 {py.sum()/st.sum()*100:.0f}% [{np.percentile(b,2.5):.0f}〜{np.percentile(b,97.5):.0f}]"
print('== 3連単 その番手が頭（10/1〜11/19）')
for lab,m in (('玉砕型(成績)・20〜100倍',o.g_stat&o.中穴),('玉砕型(成績)・全部の目',o.g_stat),('玉砕型(コメント)・20〜100倍',o.g_cmt&o.中穴),('玉砕型(コメント)・全部の目',o.g_cmt)):
    print(' ',lab,summ(o[m]))
print('== コメント型の内訳')
for per,m in (('6/11〜8/31',ban.date<'2026-09-01'),('9/1〜',ban.date>='2026-09-01')):
    print(f'   {per}:',ratio(ban[ban.薄&ban.g_cmt&m]) if (ban.薄&ban.g_cmt&m).sum()>=50 else f"{(ban.薄&ban.g_cmt&m).sum()}人 1着{int(ban[ban.薄&ban.g_cmt&m].won.sum())}回 / 見込み{ban[ban.薄&ban.g_cmt&m].q.sum():.1f}回")
x=ban[ban.薄&ban.g_cmt]
print('   番手の人気 4〜6番:',ratio(x[x['rank']<=6]) ,' 7番以下:',f"{(x['rank']>6).sum()}人 1着{int(x[x['rank']>6].won.sum())}回 見込み{x[x['rank']>6].q.sum():.1f}")
print('   先頭のコメント例:',lead[lead.g_cmt].comment.value_counts().head(8).to_dict())
# 2車単 番手→（誰でも）の回収率: 実際の2車単の払戻
pay=pd.read_sql("select race_id,combo,payout from payouts where bet_type='2車単'",c); P2=dict(zip(pay.race_id,pay.combo)); V2=dict(zip(pay.race_id,pay.payout))
def nagashi(d):
    st=0; py=0
    for r,h in zip(d.race_id,d.car_no):
        n=int(D[D.race_id==r].shape[0])-1; st+=n*100; w=P2.get(r,'')
        if w.split('>')[0]==str(int(h)): py+=V2[r]
    return f"{len(d)}件 回収率 {py/st*100:.0f}%"
print('   2車単 番手→全員に流す（コメント型・人気薄の番手）:',nagashi(x))
