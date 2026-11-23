import pandas as pd, numpy as np, sqlite3
d=pd.read_pickle('odds_val.pkl'); d=d[d.method=='LightGBM+競走得点']
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
e=pd.read_sql("select race_id,car_no,line_id,line_pos,line_size from entries where scratched=0",c)
E={(r.race_id,int(r.car_no)):(r.line_id,r.line_pos,r.line_size) for r in e.itertuples()}
def rel(rid,cb):
    a,b=[int(x) for x in cb.replace('>','-').split('-')[:2]]
    la,lb=E.get((rid,a)),E.get((rid,b))
    if not la or not lb: return '不明'
    if la[0]==lb[0] and la[2]>=2:
        ps=sorted([la[1],lb[1]])
        return '同ライン'+('先頭-番手' if ps==[1,2] else '番手-3番手' if ps==[2,3] else 'その他')
    return '別ライン'
pd.set_option('display.width',250)
for bt in ['2車複','ワイド','2車単']:
    x=d[d.bt==bt].copy(); x['rel']=[rel(r,cb) for r,cb in zip(x.race_id,x.combo)]
    x['hit']=(x.ret>0).astype(float)
    g=x.groupby('rel').agg(点=('hit','size'),実際=('hit','sum'),AI予測=('p','sum'),市場予測=('q','sum'),払戻=('ret','sum'))
    g['実際/AI']=(g.実際/g.AI予測).round(2); g['実際/市場']=(g.実際/g.市場予測).round(2); g['全部買い回収%']=(g.払戻/g.点).round(1)
    print('==',bt); print(g[['点','実際','実際/AI','実際/市場','全部買い回収%']].to_string())
    x.to_pickle(f'line_{bt}.pkl')
