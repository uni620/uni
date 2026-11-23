"""ラインが崩れる（1着と2着が別ライン）を、レース前の材料で予想できるか。材料のグループごとに効き目を測る。"""
import pandas as pd,numpy as np,lightgbm as lgb,warnings,sqlite3
warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
E=pd.read_pickle(f'{S}/hx2.pkl')
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
AR=pd.read_pickle(f'{S}/ana_R.pkl').set_index('race_id')
E=E[E.race_id.isin(AR.index)].copy()
# --- 今節の前の日の動き（同じ場で1〜3日前の直近のレース）
E['dt']=pd.to_datetime(E.date); E=E.sort_values(['rider_id','dt'])
g=E.groupby('rider_id')
for col,new in (('finish','前走着'),('gotB','前走B'),('dt','前走日'),('venue','前走場'),('先頭','前走先頭')):
    E[new]=g[col].shift(1)
same=(E.前走場==E.venue)&((E.dt-E.前走日).dt.days<=3)
E['今節前走あり']=same.astype(float)
for k in ('前走着','前走B','前走先頭'): E['今節'+k]=E[k].where(same)
# 前走でちぎられた（番手以降で、前走の自分の先頭が3着以内なのに自分は着外）→ 近似: 前走着>=5 かつ 前走は番手以降
E['今節前走大敗']=((E.今節前走着>=6)).astype(float).where(same)
# --- レース単位に集計
R=AR[['並び','グレード','p3']].copy()
W=E[E.finish==1].drop_duplicates('race_id').set_index('race_id'); S2=E[E.finish==2].drop_duplicates('race_id').set_index('race_id')
R['崩れ']=(W.lkey.reindex(R.index)!=S2.lkey.reindex(R.index)).astype(float)
R.loc[W.lkey.reindex(R.index).isna()|S2.lkey.reindex(R.index).isna(),'崩れ']=np.nan
R['20倍以上']=(R.p3>=2000).astype(float)
G=E.groupby('race_id')
ld=E[(E.先頭==1)|(E.単騎==1)]; LG=ld.groupby('race_id')
F=pd.DataFrame(index=R.index)
# 基本
F['頭数']=G.size(); F['ライン数']=G.lkey.nunique(); F['単騎数']=G.単騎.sum(); F['最長ライン']=G.ライン人数.max()
F['3車以上ライン数']=E[E.先頭==1].groupby('race_id').ライン人数.apply(lambda s:(s>=3).sum())
F['級']=G.race_class_code.first() if 'race_class_code' in E else 0
F['グレード']=R.グレード.map({'F2':0,'F1':1,'G3以上':2}); F['周長']=G.周長.first(); F['種別']=G.種別.first()
BASE=list(F.columns)
# 1 先行争い
F['先行候補数']=G.先行候補数.first(); F['逃率上位2差']=LG.逃率.apply(lambda s:(s.nlargest(2).iloc[0]-s.nlargest(2).iloc[-1]) if len(s)>1 else 1)
F['B率上位2差']=LG.B率.apply(lambda s:(s.nlargest(2).iloc[0]-s.nlargest(2).iloc[-1]) if len(s)>1 else 1)
F['逃率合計']=LG.逃率.sum(); F['pB最大']=G.pB.max(); F['pB上位2差']=G.pB.apply(lambda s:s.nlargest(2).iloc[0]-s.nlargest(2).iloc[-1]); F['競り合い']=G.競り合い.first()
SEN=['先行候補数','逃率上位2差','B率上位2差','逃率合計','pB最大','pB上位2差','競り合い']
# 2 ラインの中身
F['得点差1-2']=G.score.apply(lambda s:s.nlargest(2).iloc[0]-s.nlargest(2).iloc[-1]); F['得点の散らばり']=G.score.std()
F['得点上位2人が同ライン']=G.apply(lambda d:float(d.nlargest(2,'score').lkey.nunique()==1))
F['先頭−番手 最小']=G['先頭−番手の得点'].min(); F['先頭−番手 最大']=G['先頭−番手の得点'].max()
F['ライン得点差1-2']=G.L得点.apply(lambda s:np.sort(s.unique())[-1]-np.sort(s.unique())[-2] if s.nunique()>1 else 0)
F['強いラインのpB']=G.apply(lambda d:d.loc[d.L得点.idxmax(),'自分のラインのpB'])
LIN=['得点差1-2','得点の散らばり','得点上位2人が同ライン','先頭−番手 最小','先頭−番手 最大','ライン得点差1-2','強いラインのpB']
# 3 コメント
F['自力宣言数']=G.R_自力宣言数.first(); F['勝負宣言数']=G.R_勝負宣言数.first(); F['自在宣言数']=G.cm_自在.sum(); F['単騎宣言数']=G.cm_単騎.sum()
F['他線名指し数']=G.cm_他線名指し.sum(); F['不調コメ数']=G.cm_不調.sum(); F['好調コメ数']=G.cm_好調.sum(); F['コメなし数']=G.cm_なし.sum()
CMT=['自力宣言数','勝負宣言数','自在宣言数','単騎宣言数','他線名指し数','不調コメ数','好調コメ数','コメなし数']
# 4 今節の前の日
F['今節前走B数']=G.今節前走B.sum(); F['今節前走大敗数']=G.今節前走大敗.sum(); F['今節前走あり数']=G.今節前走あり.sum()
F['強い先頭の今節前走着']=G.apply(lambda d:d.loc[d.L得点.idxmax(),'今節前走着'] if d.L得点.notna().any() else np.nan)
F['得点1位の今節前走着']=G.apply(lambda d:d.loc[d.score.idxmax(),'今節前走着'])
PRV=['今節前走B数','今節前走大敗数','今節前走あり数','強い先頭の今節前走着','得点1位の今節前走着']
# 5 ラインの過去
F['強いラインの連係ワンツー率']=G.apply(lambda d:d.loc[d.L得点.idxmax(),'連係時のワンツー率'] if d.L得点.notna().any() else np.nan)
F['番手ちぎられ率最大']=E[E.番手==1].groupby('race_id').ちぎられ率.max(); F['先頭ちぎり率最大']=E[E.先頭==1].groupby('race_id').ちぎり率.max()
F['先頭のB時平均着最小']=E[E.先頭==1].groupby('race_id').B時の平均着順.min()
HIS=['強いラインの連係ワンツー率','番手ちぎられ率最大','先頭ちぎり率最大','先頭のB時平均着最小']
D=F.join(R[['崩れ','20倍以上']]); D['date']=E.groupby('race_id').date.first()
D.to_pickle(f'{S}/breakfeat.pkl')
from sklearn_free_auc import auc
