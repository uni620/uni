"""3連単の全組み合わせの確率を、頭AI（上位3人の頭）×展開型の相手AIで出す。9〜11月、各月その月より前で学習。"""
from hx_lib import *
import sqlite3, itertools
M=[('2026-09-01','2026-10-01'),('2026-10-01','2026-11-01'),('2026-11-01','2026-11-21')]
src=open(f'{S}/hx_topp.py').read()
exec(src[src.index('E=pd.read_pickle'):src.index('c=sqlite3.connect')])     # E, base, relto, RH, RB, PBIN
out=[]
for a,b in M:
    x=E[E.date<a]; y=E[(E.date>=a)&(E.date<b)].copy()
    y['pr']=sum(lgb.train(dict(PBIN,seed=s),lgb.Dataset(x[base],x.win),500).predict(y[base]) for s in (1,2))
    y['pr']/=y.groupby('race_id').pr.transform('sum')
    hw=x[x.win==1].set_index('race_id').car_no; bw=x[x.gotB==1].set_index('race_id').car_no
    X1=relto(x.copy(),hw,'頭'); m1=lgb.train(PBIN,lgb.Dataset(X1[base+RH],X1.gotB),400)
    X2=relto(relto(x[x.win==0].copy(),hw,'頭'),bw,'B')
    m2=lgb.train(PBIN,lgb.Dataset(X2[base+RH+RB],(X2.finish==2).astype(int)),400)
    m3=lgb.train(PBIN,lgb.Dataset(X2[base+RH+RB],(X2.finish==3).astype(int)),400)
    ys=y.sort_values(['race_id','pr'],ascending=[True,False])
    for hk in range(3):
        hh=ys.groupby('race_id').nth(hk).set_index('race_id')
        Y1=relto(y.copy(),hh.car_no,'頭'); Y1['pb']=m1.predict(Y1[base+RH]); Y1['pb']/=Y1.groupby('race_id').pb.transform('sum')
        cand=Y1.sort_values(['race_id','pb'],ascending=[True,False]).groupby('race_id').head(3)
        others=y[y.car_no!=y.race_id.map(hh.car_no)]
        parts=[]
        for k in range(3):
            bk=cand.groupby('race_id').nth(k).set_index('race_id')
            Z=relto(relto(others[others.race_id.isin(bk.index)].copy(),hh.car_no,'頭'),bk.car_no,'B')
            Z['w']=Z.race_id.map(bk.pb); Z['p2']=m2.predict(Z[base+RH+RB]); Z['p3']=m3.predict(Z[base+RH+RB])
            for col in ('p2','p3'): Z[col]=Z[col]/Z.groupby('race_id')[col].transform('sum')*Z.w
            parts.append(Z[['race_id','car_no','p2','p3']])
        Z=pd.concat(parts).groupby(['race_id','car_no'])[['p2','p3']].sum().reset_index()
        Z['head']=Z.race_id.map(hh.car_no); Z['hpr']=Z.race_id.map(hh.pr)
        out.append(Z)
    print(a[:7],'done',flush=True)
Z=pd.concat(out); Z.to_pickle(f'{S}/tri_full_Z.pkl')
# 3連単の確率表
rows=[]
for (rid,h),g in Z.groupby(['race_id','head']):
    q2=(g.set_index('car_no').p2/g.p2.sum()); q3=(g.set_index('car_no').p3/g.p3.sum()); hp=g.hpr.iloc[0]
    for u in q2.index:
        rest=q3.drop(u); rest=rest/rest.sum()
        for v in rest.index: rows.append((rid,f'{int(h)}>{int(u)}>{int(v)}',hp*q2[u]*rest[v]))
T=pd.DataFrame(rows,columns=['race_id','combo','P']); T.to_pickle(f'{S}/tri_full.pkl'); print(len(T))
