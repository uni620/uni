from hx_lib import *
E=pd.read_pickle(f'{S}/hx2.pkl'); C=pickle.load(open(f'{S}/hx_cols.pkl','rb'))
PFC,CC=C['PFC'],C['CC']; FEAT=[f for f in C['FEAT'] if f not in PFC]; base=FEAT+CC
P=dict(objective='binary',learning_rate=0.03,num_leaves=15,min_data_in_leaf=200,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,verbose=-1)
out=[]
for a,b in MONTHS:
    x=E[E.date<a]; y=E[(E.date>=a)&(E.date<b)].copy(); p=0
    for sd in (1,2): p=p+lgb.train(dict(P,seed=sd),lgb.Dataset(x[base],x.win),500).predict(y[base])/2
    y['pr']=p; y['pr']/=y.groupby('race_id').pr.transform('sum'); out.append(y[['race_id','date','car_no','win','pr']])
Y=pd.concat(out).merge(pd.read_pickle(f'{S}/hx_Y.pkl')[['race_id','car_no','p','q']],on=['race_id','car_no'])
Y.to_pickle(f'{S}/hx_conf.pkl'); t=Y.loc[Y.groupby('race_id').pr.idxmax()].set_index('race_id'); mk=Y.loc[Y.groupby('race_id').q.idxmax()].set_index('race_id'); t['mk']=mk.win
print('確率AI 本命1着率',round(t.win.mean()*100,2))
t['帯']=pd.cut(t.pr,[0,.3,.4,.5,.6,1])
r=t.groupby('帯',observed=True).agg(R=('win','size'),AIの見込み=('pr','mean'),実際=('win','mean'),人気1番=('mk','mean')); print((r*[1,100,100,100]).round(1))
for lab,m in (('9月',t.date<'2026-10-01'),('10月',(t.date>='2026-10-01')&(t.date<'2026-11-01')),('11月',t.date>='2026-11-01')):
    x=t[m]; print(lab,'見込み50%以上',len(x[x.pr>=.5]),'R',round(x[x.pr>=.5].win.mean()*100,1),'% | 60%以上',len(x[x.pr>=.6]),'R',round(x[x.pr>=.6].win.mean()*100,1),'%')
