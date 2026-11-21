"""手順3の準備: レースごとに 2着・3着の候補ごとの特徴を作る（すべてそのレースより前の日のデータだけ）。"""
import sqlite3, numpy as np, pandas as pd, pickle
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect(f'{S}/sim.sqlite')
E=pd.read_sql("select e.race_id,e.car_no,e.rider_id,e.finish,e.line_id,e.line_pos,e.line_size,e.scratched,r.date from entries e join races r using(race_id)",c)
E['finish']=pd.to_numeric(E.finish,errors='coerce')
# --- コンビ（先頭→番手）と 番手の「先頭が勝ったとき2着」実績を日付順に累積（当日を含めない）
L=E[(E.line_size>=2)&(E.line_pos.isin([1,2]))&(E.scratched!=1)]
lead=L[L.line_pos==1][['race_id','line_id','rider_id','finish','date']].rename(columns={'rider_id':'lr','finish':'lf'})
fol=L[L.line_pos==2][['race_id','line_id','rider_id','finish']].rename(columns={'rider_id':'fr','finish':'ff'})
PR=lead.merge(fol,on=['race_id','line_id']).dropna(subset=['lf','ff']).sort_values('date')
PR['both12']=((PR.lf<=2)&(PR.ff<=2)).astype(float)
PR['lw']=(PR.lf==1).astype(float); PR['fw2']=((PR.lf==1)&(PR.ff==2)).astype(float)
def cum_before(df,keys,cols):
    g=df.groupby(keys+['date'])[cols].sum().groupby(level=list(range(len(keys)))).cumsum()
    g=g.groupby(level=list(range(len(keys)))).shift(1).fillna(0)   # その日より前まで
    return g
pair=cum_before(PR.assign(n=1.0),['lr','fr'],['n','both12'])
folw=cum_before(PR,['fr'],['lw','fw2'])
base12=PR.both12.mean(); basef=PR.fw2.sum()/PR.lw.sum()
print('全体: コンビで1-2独占', round(base12,3), '先頭1着時に番手2着', round(basef,3))
pickle.dump(dict(pair=pair,folw=folw,base12=base12,basef=basef),open(f'{S}/step3_hist.pkl','wb'))
