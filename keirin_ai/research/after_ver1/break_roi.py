import pandas as pd,numpy as np,lightgbm as lgb,warnings
warnings.filterwarnings('ignore')
exec(open('breakfeat_eval.py').read().split("for target in")[0])
ALL=BASE+SEN+LIN+CMT+PRV+HIS
out=[]
for a,b in zip(MON[:-1],MON[1:]):
    tr=D[(D.date<a)&D['崩れ'].notna()]; te=D[(D.date>=a)&(D.date<b)].copy()
    te['p崩れ']=lgb.train(P,lgb.Dataset(tr[ALL],tr['崩れ']),300).predict(te[ALL]); out.append(te[['p崩れ']])
PB=pd.concat(out)
X=pd.read_pickle('suji_chigai.pkl').join(PB,on='race_id')
X['崩れ予想']=pd.qcut(X.groupby('買い方').p崩れ.rank(pct=True),[0,.2,.4,.6,.8,1],labels=['低い20%','2','3','4','高い20%'])
t=X.groupby(['買い方','崩れ予想'],observed=True).apply(lambda d:pd.Series({'的中':round((d.ret>0).mean()*100,1),'回収':round(d.ret.sum()/d.cost.sum()*100,1)}))
pd.set_option('display.width',250); print(t.unstack('崩れ予想').to_string())
