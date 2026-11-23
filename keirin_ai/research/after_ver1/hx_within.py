from hx_lib import *
E=pd.read_pickle(f'{S}/hx2.pkl'); C=pickle.load(open(f'{S}/hx_cols.pkl','rb'))
Y=pd.read_pickle(f'{S}/hx_Y.pkl')[['race_id','car_no','q']]
E=E.merge(Y,on=['race_id','car_no'],how='left')
RF=['score','win_rate','ren2','ren3','逃率','捲率','差率','マ率','B率','S率','B_1着率','B_3着内率','先_1着率','先_3着内率','先_回数','番_1着率','番_3着内率','番_回数',
    '近5_agari_pct','近5_behind','近5_fin_pct','内容−着順','cm_自力','cm_勝負','cm_自在','cm_付く','cm_好調','cm_不調','cm_味方名指し','cm_他線名指し',
    'pf_win_3m','pf_top3_3m','pf_recent_finish','pf_trend','pf_kaku_now','pf_kaku_diff','通算1着率','通算3着内率','普段の先頭率','普段の番手率','得点の伸び(10走)','age']
RF=[f for f in RF if f in E]
L=E[E.先頭==1].set_index('lkey'); B=E[E.番手==1].set_index('lkey')
k=L.index.intersection(B.index)
L=L.loc[k]; B=B.loc[k]
X=pd.DataFrame(index=k)
for f in RF: X['先_'+f]=L[f].values; X['番_'+f]=B[f].values; X['差_'+f]=L[f].values-B[f].values
for f in ['ライン人数','ライン数','先行候補数','n','pB','自分のラインのpB','他ラインの最大pB','競り合い','R_自力宣言数','R_勝負宣言数','連係回数','連係時の先頭3着内率','連係時のワンツー率','種別','グレード','周長']:
    if f in L: X['L_'+f]=L[f].values
X['date']=L.date.values; X['y']=L.win.values; X['either']=(L.win.values+B.win.values)
X['q先']=L.q.values; X['q番']=B.q.values
X=X[X.either==1]
F=[c for c in X.columns if c not in('date','y','either','q先','q番')]
P=dict(objective='binary',learning_rate=0.03,num_leaves=15,min_data_in_leaf=100,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,verbose=-1)
acc=[];mk=[];sc=[];n=0;rows=[]
for a,b in MONTHS:
    tr=X[X.date<a]; te=X[(X.date>=a)&(X.date<b)]
    p=np.zeros(len(te))
    for sd in (1,2,3): p+=lgb.train(dict(P,seed=sd),lgb.Dataset(tr[F],tr.y),400).predict(te[F])/3
    ok=te.q先.notna()&te.q番.notna()
    acc.append(((p>0.5)==te.y)[ok].sum()); mk.append(((te.q先>te.q番)==te.y)[ok].sum()); sc.append(((te.差_score>0)==te.y)[ok].sum()); n+=ok.sum()
    rows.append(te.assign(pin=p))
print('ラインが勝ったときの先頭/番手の判定（9〜11月）',n,'本')
print(' ライン内AI',round(sum(acc)/n*100,1),'% / 人気',round(sum(mk)/n*100,1),'% / 得点が上の方',round(sum(sc)/n*100,1),'%')
m=lgb.train(dict(P,seed=1),lgb.Dataset(X[F],X.y),400); imp=pd.Series(m.feature_importance('gain'),index=F)
print('効いた特徴',(imp/imp.sum()).sort_values(ascending=False).head(12).round(3).to_dict())
pd.concat(rows).to_pickle(f'{S}/hx_within_pred.pkl')
