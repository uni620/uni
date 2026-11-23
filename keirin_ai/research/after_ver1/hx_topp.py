"""展開型の相手AI: 頭（1着）を出発点に、段階1=頭が勝つときに誰がBを取るか、段階2=その展開で誰が2着・3着か。
Bの候補上位3人で重み付けして合わせる。4〜11月、各月その月より前だけで学習。今の相手AIと同じ買い方で比べる。"""
from hx_lib import *
import sqlite3, itertools
M=[('2026-04-01','2026-05-01'),('2026-05-01','2026-06-01'),('2026-06-01','2026-07-01'),('2026-07-01','2026-08-01'),('2026-08-01','2026-09-01'),
   ('2026-09-01','2026-10-01'),('2026-10-01','2026-11-01'),('2026-11-01','2026-11-21')]
E=pd.read_pickle(f'{S}/hx2.pkl'); C=pickle.load(open(f'{S}/hx_cols.pkl','rb'))
PFC,CC=C['PFC'],C['CC']; FEAT=[f for f in C['FEAT'] if f not in PFC]; base=FEAT+CC
E=E[E.groupby('race_id').gotB.transform('sum')==1].copy()
key=E.set_index(['race_id','car_no'])
def relto(X,ref,pfx):
    """ref: race_id -> car_no。X の各選手と ref の選手の関係。"""
    r=key.loc[list(zip(ref.index,ref.values))].reset_index().set_index('race_id')
    same=(X.lkey==X.race_id.map(r.lkey)); d=X.line_pos-X.race_id.map(r.line_pos)
    X[pfx+'と同ライン']=same.astype(float); X[pfx+'の直後']=(same&(d==1)).astype(float); X[pfx+'の直前']=(same&(d==-1)).astype(float)
    X[pfx+'の2つ後ろ']=(same&(d==2)).astype(float)
    X[pfx+'が先頭']=X.race_id.map(r.先頭); X[pfx+'が番手']=X.race_id.map(r.番手); X[pfx+'が単騎']=X.race_id.map(r.単騎)
    X[pfx+'との得点差']=X.score-X.race_id.map(r.score)
    return X
RH=[f'頭{s}' for s in ('と同ライン','の直後','の直前','の2つ後ろ','が先頭','が番手','が単騎','との得点差')]
RB=[f'B{s}' for s in ('と同ライン','の直後','の直前','の2つ後ろ','が先頭','が番手','が単騎','との得点差')]
PBIN=dict(objective='binary',learning_rate=0.03,num_leaves=15,min_data_in_leaf=200,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,verbose=-1)
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
pay=pd.read_sql("select race_id,bet_type,combo,payout from payouts",c); PAY={(r,b,k):v for r,b,k,v in pay.itertuples(index=False)}
rows=[]; acc=[]
for a,b in M:
    x=E[E.date<a]; y=E[(E.date>=a)&(E.date<b)].copy()
    # 頭AI（確率）
    p=sum(lgb.train(dict(PBIN,seed=s),lgb.Dataset(x[base],x.win),500).predict(y[base]) for s in (1,2))
    y['pr']=p; y['pr']/=y.groupby('race_id').pr.transform('sum')
    hh=y.loc[y.groupby('race_id').pr.idxmax()].set_index('race_id')
    # 学習: 実際の1着を頭に
    hw=x[x.win==1].set_index('race_id').car_no; bw=x[x.gotB==1].set_index('race_id').car_no
    X1=relto(x.copy(),hw,'頭')                          # 段階1: 頭が勝つとき誰がBか（頭自身も候補）
    m1=lgb.train(PBIN,lgb.Dataset(X1[base+RH],X1.gotB),400)
    X2=relto(relto(x[x.win==0].copy(),hw,'頭'),bw,'B')  # 段階2: 頭とBが決まったとき 2着/3着
    m2=lgb.train(PBIN,lgb.Dataset(X2[base+RH+RB],(X2.finish==2).astype(int)),400)
    m3=lgb.train(PBIN,lgb.Dataset(X2[base+RH+RB],(X2.finish==3).astype(int)),400)
    # 確認: 頭AIの頭で
    Y1=relto(y.copy(),hh.car_no,'頭'); Y1['pb']=m1.predict(Y1[base+RH]); Y1['pb']/=Y1.groupby('race_id').pb.transform('sum')
    cand=Y1.sort_values(['race_id','pb'],ascending=[True,False]).groupby('race_id').head(3)
    others=y[y.car_no!=y.race_id.map(hh.car_no)]
    acc2=None; parts=[]
    for k in range(3):
        bk=cand.groupby('race_id').nth(k).set_index('race_id')
        w=bk.pb; Z=relto(relto(others[others.race_id.isin(bk.index)].copy(),hh.car_no,'頭'),bk.car_no,'B')
        Z['w']=Z.race_id.map(w); Z['p2']=m2.predict(Z[base+RH+RB]); Z['p3']=m3.predict(Z[base+RH+RB])
        for col in ('p2','p3'): Z[col]=Z[col]/Z.groupby('race_id')[col].transform('sum')*Z.w
        parts.append(Z[['race_id','car_no','finish','p2','p3']])
    Z=pd.concat(parts).groupby(['race_id','car_no']).agg(finish=('finish','first'),p2=('p2','sum'),p3=('p3','sum')).reset_index()
    o2=Z.sort_values(['race_id','p2'],ascending=[True,False]).groupby('race_id').car_no.apply(list)
    o3=Z.assign(p23=Z.p2+Z.p3).sort_values(['race_id','p23'],ascending=[True,False]).groupby('race_id').car_no.apply(list)
    # 段階1の当たり（頭が当たったレースで、Bの1位が実際のB）
    hit=hh[hh.win==1].index; c1=cand.groupby('race_id').head(1).set_index('race_id')
    acc.append((c1.loc[c1.index.intersection(hit)].gotB.mean()))
    for rid,h in hh.iterrows():
        if rid not in o2.index: continue
        hd=int(h.car_no); s2=o2[rid]; s3=o3[rid]
        sec3=s2[:3]; thr4=s3[:4] if True else None
        P={'2車単 1点':[('2車単',f'{hd}>{s2[0]}')],'2車単 3点':[('2車単',f'{hd}>{z}') for z in sec3],
           '3連複 3点':[('3連複','-'.join(map(str,sorted((hd,u,v))))) for u,v in itertools.combinations(s3[:3],2)],
           '3連単 6点':[('3連単',f'{hd}>{u}>{v}') for u in s2[:2] for v in [t for t in s3 if t!=u][:3]],
           '3連単 9点':[('3連単',f'{hd}>{u}>{v}') for u in sec3 for v in [t for t in s3 if t!=u][:3]]}
        for nm,bt in P.items(): rows.append((a[:7],rid,h.pr,h.win,nm,len(bt)*100,sum(PAY.get((rid,t,k),0) for t,k in bt)))
    print(a[:7],'done 段階1(B)の的中',round(acc[-1]*100,1),flush=True)
R=pd.DataFrame(rows,columns=['月','race_id','pr','headwin','買い方','cost','ret']); R.to_pickle(f'{S}/hx_topp.pkl')
