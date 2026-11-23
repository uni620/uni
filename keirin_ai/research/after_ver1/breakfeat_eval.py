import pandas as pd,numpy as np,lightgbm as lgb,warnings,sys
warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
from sklearn_free_auc import auc
D=pd.read_pickle('breakfeat.pkl')
src=open('breakfeat.py').read()
G={}
for name in ('BASE','SEN','LIN','CMT','PRV','HIS'):
    pass
BASE=['頭数','ライン数','単騎数','最長ライン','3車以上ライン数','級','グレード','周長','種別']
SEN=['先行候補数','逃率上位2差','B率上位2差','逃率合計','pB最大','pB上位2差','競り合い']
LIN=['得点差1-2','得点の散らばり','得点上位2人が同ライン','先頭−番手 最小','先頭−番手 最大','ライン得点差1-2','強いラインのpB']
CMT=['自力宣言数','勝負宣言数','自在宣言数','単騎宣言数','他線名指し数','不調コメ数','好調コメ数','コメなし数']
PRV=['今節前走B数','今節前走大敗数','今節前走あり数','強い先頭の今節前走着','得点1位の今節前走着']
HIS=['強いラインの連係ワンツー率','番手ちぎられ率最大','先頭ちぎり率最大','先頭のB時平均着最小']
P=dict(objective='binary',learning_rate=0.03,num_leaves=15,min_data_in_leaf=100,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,seed=1)
MON=[f'2026-{m:02d}-01' for m in range(4,12)]+['2026-11-21']
def wf(F,target):
    ys=[];ps=[]
    for a,b in zip(MON[:-1],MON[1:]):
        tr=D[(D.date<a)&D[target].notna()]; te=D[(D.date>=a)&(D.date<b)&D[target].notna()]
        m=lgb.train(P,lgb.Dataset(tr[F],tr[target]),300); ps.append(m.predict(te[F])); ys.append(te[target].values)
    y=np.concatenate(ys); p=np.concatenate(ps); return auc(y,p),y,p
for target in ('崩れ','20倍以上'):
    print(f'=== 目標: {target}（全体の割合 {D[target].mean()*100:.1f}%）')
    a0,_,_=wf(BASE,target); print(f'  基本だけ AUC {a0:.3f}')
    for nm,X in (('+先行争い',SEN),('+ラインの中身',LIN),('+コメント',CMT),('+今節の前の日',PRV),('+ラインの過去',HIS)):
        a,_,_=wf(BASE+X,target); print(f'  {nm} AUC {a:.3f}（{a-a0:+.3f}）')
    a,y,p=wf(BASE+SEN+LIN+CMT+PRV+HIS,target); print(f'  全部 AUC {a:.3f}（{a-a0:+.3f}）')
    q=pd.qcut(p,5,labels=['低い20%','2','3','4','高い20%']); print('  全部の予想の5段階ごとの実際の割合',pd.Series(y).groupby(q).mean().mul(100).round(1).to_dict())
    m=lgb.train(P,lgb.Dataset(D[D[target].notna()][BASE+SEN+LIN+CMT+PRV+HIS],D[D[target].notna()][target]),300)
    imp=pd.Series(m.feature_importance('gain'),index=BASE+SEN+LIN+CMT+PRV+HIS); print('  効いた材料',(imp/imp.sum()).sort_values(ascending=False).head(10).round(3).to_dict())
