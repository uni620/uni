"""勝負レース選びAI: 頭AIの本命が1着になるか を、頭AIの自信＋レースの条件で予想（月ごとに学習し直し, 4〜11月の結果を使い 6〜11月を評価）。"""
import pandas as pd,numpy as np,lightgbm as lgb,warnings,sys
warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
V=pd.read_pickle('hx_topp.pkl')
P3=V[V.買い方=='2車単 3点'][['race_id','月','pr','headwin','cost','ret']].set_index('race_id')
T9=V[V.買い方=='3連単 9点'].set_index('race_id')[['cost','ret']].rename(columns={'cost':'c9','ret':'r9'})
D=pd.read_pickle('breakfeat.pkl'); AR=pd.read_pickle('ana_R.pkl').set_index('race_id')
X=P3.join(T9).join(D.drop(columns=['崩れ','20倍以上','date'],errors='ignore')).join(AR[['日程','時間帯']])
X['日程']=X.日程.map({'初日':0,'中日':1,'最終日':2}); X['時間帯']=X.時間帯.map({'モーニング':0,'デイ':1,'ナイター':2,'ミッドナイト':3}).astype(float)
F=['pr']+[c for c in D.columns if c not in ('崩れ','20倍以上','date')]+['日程','時間帯']
P=dict(objective='binary',learning_rate=0.03,num_leaves=7,min_data_in_leaf=100,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,seed=1)
out=[]
months=sorted(X.月.unique())
for m in months[2:]:
    tr=X[X.月<m]; te=X[X.月==m].copy()
    te['meta']=lgb.train(P,lgb.Dataset(tr[F],tr.headwin),300).predict(te[F]); out.append(te)
Y=pd.concat(out); print('評価',Y.月.min(),'〜',Y.月.max(),len(Y),'R')
for top in (0.02,0.05,0.08,0.1,0.15,0.2):
    k=int(len(Y)*top)
    a=Y.nlargest(k,'pr'); b=Y.nlargest(k,'meta')
    f=lambda d:f"頭{d.headwin.mean()*100:.1f}% 2車単3点{(d.ret>0).mean()*100:.1f}%(回収{d.ret.sum()/d.cost.sum()*100:.0f}%) 3連単9点{(d.r9>0).mean()*100:.1f}%"
    print(f'上位{top*100:.0f}%({k}R,{k/len(Y.index.str[:8].unique())*1:.1f}R/日) 自信だけ: {f(a)} | 選びAI: {f(b)}')
