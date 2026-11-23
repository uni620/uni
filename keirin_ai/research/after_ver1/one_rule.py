"""回収率を上げる1つのルール探し（事前登録）。
モデル: 券種ごとに LightGBM（組み合わせが当たるか）。材料 = オッズのずれ（ver1の特徴）＋オッズ＋選手の材料（位置・ライン・主導権・得点順位・状況別成績・調子）。
評価: 8〜11月、各月その月より前で学習。比較 = ver1 と同じロジスティック（オッズのずれだけ）を同じ条件で学習。
事前登録の買い方: 2車複・2車単・3連複から、10倍以下で期待値最大の1点、期待値>=1.1なら買う（ワイドなし）。"""
import pandas as pd,numpy as np,lightgbm as lgb,warnings
from scipy.optimize import minimize
warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
T=pd.read_pickle(f'{S}/step5_T.pkl'); T=T[T.bt!='ワイド'].copy()
E=pd.read_pickle(f'{S}/hx2.pkl')
RF=['pB','自分のラインのpB','先頭','番手','単騎','得点順位','B率','逃率','先_1着率','番_1着率','近5_fin_pct','近5_agari_pct','score','ライン人数']
K=E.set_index(['race_id','car_no'])[RF+['lkey']]
cars=T.combo.str.replace('>','-').str.split('-')
T['c1']=cars.str[0].astype(int); T['c2']=cars.str[1].astype(int); T['c3']=cars.str[2].fillna(0).astype(int)
def get(cn,col):
    idx=pd.MultiIndex.from_arrays([T.race_id,cn]); return K[col].reindex(idx).values
for i in (1,2,3):
    for col in ['pB','先頭','番手','単騎','得点順位','先_1着率','番_1着率','近5_fin_pct','lkey']:
        T[f'{col}_{i}']=get(T[f'c{i}'],col) if i<3 else np.where(T.c3>0,get(T.c3,col),np.nan)
T['同ライン12']=(T.lkey_1==T.lkey_2).astype(float); T['同ライン全']=((T.lkey_1==T.lkey_2)&((T.c3==0)|(T.lkey_2==T.lkey_3))).astype(float)
for col in ['pB','得点順位','近5_fin_pct']:
    T[col+'_min']=T[[f'{col}_{i}' for i in (1,2,3)]].min(axis=1); T[col+'_max']=T[[f'{col}_{i}' for i in (1,2,3)]].max(axis=1)
T['先頭数']=T[['先頭_1','先頭_2','先頭_3']].sum(axis=1); T['番手数']=T[['番手_1','番手_2','番手_3']].sum(axis=1); T['単騎数']=T[['単騎_1','単騎_2','単騎_3']].sum(axis=1)
for col in ['q','q_ex','q_tr','q_qu','q_wi','share','s_wi','s_qu','s_ex','w3','odds']: T['l'+col]=np.log(T[col].astype(float).clip(lower=1e-9))
T['hit']=(T.pay>0).astype(int)
MK=['lq','lq_ex','lq_tr','lq_qu','lq_wi','lshare','ls_wi','ls_qu','ls_ex','lw3','lodds','nine']
RID=['pB_1','pB_2','pB_3','先頭_1','先頭_2','番手_1','番手_2','単騎_1','単騎_2','得点順位_1','得点順位_2','得点順位_3','先_1着率_1','番_1着率_1','番_1着率_2',
     '近5_fin_pct_1','近5_fin_pct_2','同ライン12','同ライン全','pB_min','pB_max','得点順位_min','得点順位_max','近5_fin_pct_min','先頭数','番手数','単騎数','hour']
T.to_pickle(f'{S}/one_rule_T.pkl')
COLS={'2車複':['lq','lq_ex','lq_tr','lq_qu','lq_wi'],'2車単':['lq','lq_ex','lq_tr','lq_qu','lq_wi','lshare'],'3連複':['lq','ls_wi','ls_qu','ls_ex','lw3']}
def lr_fit(X,y,C=1.0):
    mu=X.mean(0); sd=X.std(0)+1e-9; Z=np.column_stack([np.ones(len(X)),(X-mu)/sd])
    def f(w):
        z=Z@w; p=1/(1+np.exp(-z)); return -(y*np.log(p+1e-12)+(1-y)*np.log(1-p+1e-12)).sum()+0.5/C*(w[1:]**2).sum(), Z.T@(p-y)+np.r_[0,w[1:]/C]
    w=minimize(f,np.zeros(Z.shape[1]),jac=True,method='L-BFGS-B').x
    return lambda X2:1/(1+np.exp(-(np.column_stack([np.ones(len(X2)),(X2-mu)/sd])@w)))
P=dict(objective='binary',learning_rate=0.03,num_leaves=15,min_data_in_leaf=300,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,verbose=-1,seed=1)
M=[('2026-08-01','2026-09-01'),('2026-09-01','2026-10-01'),('2026-10-01','2026-11-01'),('2026-11-01','2026-11-21')]
res=[]
for a,b in M:
    parts=[]
    for bt,cols in COLS.items():
        x=T[(T.bt==bt)&(T.date<a)].dropna(subset=cols); y=T[(T.bt==bt)&(T.date>=a)&(T.date<b)].dropna(subset=cols).copy()
        f=lambda d:np.column_stack([d[c].astype(float) for c in cols]+[d.lodds.astype(float),d.nine.astype(float)])
        y['p_lr']=lr_fit(f(x),x.hit.values.astype(float))(f(y))
        F=[c for c in MK+RID if c in T]
        m=lgb.train(P,lgb.Dataset(x[F],x.hit),600); y['p_gb']=m.predict(y[F])
        parts.append(y[['race_id','date','bt','combo','odds','pay','p_lr','p_gb']])
    Y=pd.concat(parts); Y['月']=a[:7]; res.append(Y); print(a[:7],'done',flush=True)
R=pd.concat(res); R.to_pickle(f'{S}/one_rule_R.pkl')
