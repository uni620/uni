"""手順3: 2着・3着の決め方に選手ごとの情報を足し、物差し（人気に混ぜたときの上乗せ）で比べる。
期間: 学習 6/11〜8/31 / 選択 9/1〜10/10 / 最終確認 10/11〜11/20（最終確認は手順5の最後まで使わない）。"""
import sqlite3, pickle, numpy as np, pandas as pd
from scipy.optimize import minimize
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
T1,T2='2026-09-01','2026-10-11'
c=sqlite3.connect(f'{S}/sim.sqlite')
P=pd.read_pickle(f'{S}/wf_all.pkl'); P=P[P.method=='LightGBM+競走得点'][['race_id','date','car_no','p']]
F=pd.read_pickle(f'{S}/feats_0601.pkl')[['race_id','car_no','rider_id','finish','line_id','line_pos','line_size','ren3','top3_3m','pos_top3','pos_n']]
D=P.merge(F,on=['race_id','car_no'])
H=pickle.load(open(f'{S}/step3_hist.pkl','rb'))
od=pd.read_sql("select race_id,bet_type,combo,odds,fetched_at from odds where is_final=1 and race_id>='20260611'",c)
od=od.sort_values('fetched_at').drop_duplicates(['race_id','bet_type','combo'],keep='last')
OD={k:dict(zip(g.combo,g.odds)) for k,g in od.groupby(['race_id','bet_type'])}
scr=set(pd.read_sql("select distinct race_id from entries where scratched=1",c).race_id)
# 履歴の参照（その日より前）: 累計表を日付で asof
def asof_table(cum):
    cum=cum.groupby(level=list(range(cum.index.nlevels-1))).cumsum() if False else cum
    return cum
pairc=H['pair'].reset_index(); folc=H['folw'].reset_index()   # すでに「前日まで」にずらし済み（その日の行は自分を含まない）
pairc['date']=pd.to_datetime(pairc.date); folc['date']=pd.to_datetime(folc.date)
pair_last={}
for (lr,fr),g in pairc.groupby(['lr','fr']): pair_last[(lr,fr)]=(g.date.to_numpy(),g.n.to_numpy(),g.both12.to_numpy())
fol_last={}
for fr,g in folc.groupby('fr'): fol_last[fr]=(g.date.to_numpy(),g.lw.to_numpy(),g.fw2.to_numpy())
def lookup(tab,key,d,shiftcols):
    if key not in tab: return 0.0,0.0
    ds,a,b=tab[key]; i=np.searchsorted(ds,d,side='right')-1   # 当日の行は「前日まで」の値を持つ
    return (float(a[i]),float(b[i])) if i>=0 else (0.0,0.0)
# 注: 当日の行が無い日は、それより前の最後の行（その時点の「前日まで」）→ その行の当日分が漏れないよう、行の値は前日までの累計
R=[]
for rid,g in D.groupby('race_id'):
    if rid in scr or any((rid,bt) not in OD for bt in ('2車単','3連複','2車複','ワイド')): continue
    g=g.sort_values('car_no'); n=len(g)
    if n<5: continue
    f=pd.to_numeric(g.finish,errors='coerce').to_numpy()
    cars=g.car_no.astype(int).tolist(); p=g.p.to_numpy(); p=np.clip(p/p.sum(),1e-6,1); d=np.datetime64(g.date.iloc[0])
    info=list(zip(g.line_id,g.line_pos.fillna(1),g.line_size.fillna(1)))
    rel=np.zeros((n,n,3)); own=np.zeros((n,3))
    for j,(lj,pj,sj) in enumerate(info):
        own[j]=[sj>1 and pj==2, sj>1 and pj>=3, sj<=1]
    for i,(li,pi,si) in enumerate(info):
        for j,(lj,pj,sj) in enumerate(info):
            if i==j or pd.isna(li) or pd.isna(lj) or li!=lj or si<=1: continue
            rel[i,j]=[pj==pi+1, pj==pi-1, abs(pj-pi)>1]
    def z(x):
        x=pd.to_numeric(pd.Series(x),errors='coerce').to_numpy(float); m=np.nanmean(x) if np.isfinite(x).any() else 0; s=np.nanstd(x) if np.isfinite(x).sum()>1 else 1
        return np.nan_to_num((x-m)/(s if s>0 else 1))
    place=np.column_stack([z(g.ren3),z(g.top3_3m),z(g.pos_top3)])
    rids=g.rider_id.tolist()
    fol=np.zeros(n); pr=np.zeros((n,n))
    for j in range(n):
        lw,fw=lookup(fol_last,rids[j],d,None); fol[j]=(fw+5*H['basef'])/(lw+5)-H['basef']
    for i in range(n):
        for j in range(n):
            if rel[i,j,0]:
                nn,bb=lookup(pair_last,(rids[i],rids[j]),d,None); pr[i,j]=(bb+3*H['base12'])/(nn+3)-H['base12']
    nl=len({x if pd.notna(x) else -k for k,(x,_,_) in enumerate(info)}); three=float(nl<=3); nine=float(n==9)
    lp=np.log(p)
    X2=np.zeros((n,n,14))
    X2[...,0]=lp[None,:]; X2[...,1:4]=rel; X2[...,4:7]=own[None,:,:]; X2[...,7:10]=place[None,:,:]
    X2[...,10]=rel[...,0]*fol[None,:]; X2[...,11]=rel[...,0]*pr; X2[...,12]=rel[...,0]*three; X2[...,13]=rel[...,0]*nine
    rr=np.maximum(rel[:,None,:,:],rel[None,:,:,:])            # [i,j,k,3]
    pk=np.maximum(pr[:,None,:],pr[None,:,:])
    X3=np.zeros((n,n,n,14))
    X3[...,0]=lp[None,None,:]; X3[...,1:4]=rr; X3[...,4:7]=own[None,None]; X3[...,7:10]=place[None,None]
    X3[...,10]=rr[...,0]*fol[None,None,:]; X3[...,11]=rr[...,0]*pk; X3[...,12]=rr[...,0]*three; X3[...,13]=rr[...,0]*nine
    top=[int(np.argmax(f==k)) if (f==k).sum()==1 else -1 for k in (1,2,3)]
    R.append(dict(rid=rid,date=g.date.iloc[0],cars=cars,p=p,X2=X2,X3=X3,top=top,n=n,three=three,nine=nine))
print('レース', len(R), flush=True)
pickle.dump(R,open(f'{S}/step3_R.pkl','wb'))
