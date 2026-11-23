"""相手AI: 1着（頭）が分かっているとして、残りの選手から2着・3着を当てる（オッズ・本紙なし）。
学習は実際の1着を頭にして作り、確認は頭AIの頭を使う。各月その月より前で学習。"""
from hx_lib import *
import sqlite3
E=pd.read_pickle(f'{S}/hx2.pkl'); C=pickle.load(open(f'{S}/hx_cols.pkl','rb'))
PFC,CC=C['PFC'],C['CC']; FEAT=[f for f in C['FEAT'] if f not in PFC]; base=FEAT+CC
H=pd.read_pickle(f'{S}/hx_conf.pkl')                     # 頭AI（確率）
def rel(E,head):    # head: race_id -> car_no
    X=E.copy(); X['hd']=X.race_id.map(head); X=X[X.car_no!=X.hd]
    hrow=E.set_index(['race_id','car_no']).loc[list(zip(head.index,head.values))].reset_index().set_index('race_id')
    X['頭と同じライン']=(X.lkey==X.race_id.map(hrow.lkey)).astype(float)
    hp=X.race_id.map(hrow.line_pos); X['頭の位置']=hp
    X['頭の直後']=((X.頭と同じライン==1)&(X.line_pos==hp+1)).astype(float)
    X['頭の直前']=((X.頭と同じライン==1)&(X.line_pos==hp-1)).astype(float)
    X['頭が先頭']=X.race_id.map(hrow.先頭); X['頭が番手']=X.race_id.map(hrow.番手); X['頭が単騎']=X.race_id.map(hrow.単騎)
    X['頭のライン人数']=X.race_id.map(hrow.ライン人数)
    X['頭との得点差']=X.score-X.race_id.map(hrow.score)
    X['残りでの得点順位']=X.groupby('race_id').score.rank(ascending=False)
    X['残りでの勝率順位']=X.groupby('race_id').win_rate.rank(ascending=False)
    return X
R2=['頭と同じライン','頭の位置','頭の直後','頭の直前','頭が先頭','頭が番手','頭が単騎','頭のライン人数','頭との得点差','残りでの得点順位','残りでの勝率順位']
F=base+R2
P=dict(BASEP,objective='lambdarank'); out=[]
for a,b in MONTHS:
    x=E[E.date<a]; hw=x[x.finish==1].set_index('race_id').car_no
    X=rel(x,hw); X=X.sort_values(['race_id','car_no'])
    lab=np.select([X.finish==2,X.finish==3],[2,1],0)
    d=lgb.Dataset(X[F],lab); d.set_group(X.groupby('race_id',sort=False).size().values)
    ms=[lgb.train(dict(P,seed=s),d,400) for s in (1,2)]
    y=E[(E.date>=a)&(E.date<b)]
    hh=H[(H.date>=a)&(H.date<b)]; hh=hh.loc[hh.groupby('race_id').pr.idxmax()].set_index('race_id')
    Y=rel(y[y.race_id.isin(hh.index)],hh.car_no); Y=Y.sort_values(['race_id','car_no'])
    Y['s']=sum(pd.Series(m.predict(Y[F]),index=Y.index).groupby(Y.race_id).rank() for m in ms)+1e-6*np.arange(len(Y))
    Y['hpr']=Y.race_id.map(hh.pr); Y['hwin']=Y.race_id.map(hh.win)
    out.append(Y[['race_id','date','car_no','finish','s','hpr','hwin','score','頭の直後','頭と同じライン','残りでの得点順位']])
O=pd.concat(out); O.to_pickle(f'{S}/hx_2nd.pkl')
O['rk']=O.groupby('race_id').s.rank(ascending=False,method='first')
O['帯']=pd.cut(O.hpr,[0,.5,.6,1],labels=['50%未満','50〜60%','60%以上'])
hit=O[O.hwin==1]
print('頭が当たったレースでの2着の当たり（相手AIの上位k人に2着がいる率）')
for bnd,g in hit.groupby('帯',observed=True):
    r=g[g.finish==2]; print(f' [{bnd}] {g.race_id.nunique()}R  1位 {np.mean(r.rk<=1)*100:.1f}% 2位まで {np.mean(r.rk<=2)*100:.1f}% 3位まで {np.mean(r.rk<=3)*100:.1f}%')
# 比べる基準: 残りで得点1位 / 頭の直後(番手)
g=hit; r2=g[g.finish==2]
print(' 基準: 残りの得点1位が2着',round(np.mean(r2.残りでの得点順位==1)*100,1),'% / 頭の直後の選手が2着',round(r2.頭の直後.mean()*100,1),'%（直後がいるレースは',round(g.groupby('race_id').頭の直後.max().mean()*100),'%）')
print(' 相手AI 1位が2着',round(np.mean(r2.rk==1)*100,1),'%')
# 3着: 2着が当たっている前提で3着が相手AIの上位何人に入るか
r3=g[g.finish==3]; print('3着が相手AIの上位2人に',round(np.mean(r3.rk<=2)*100,1),'% 上位3人に',round(np.mean(r3.rk<=3)*100,1),'% 上位4人に',round(np.mean(r3.rk<=4)*100,1),'%')
