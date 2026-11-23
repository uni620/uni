import sys; sys.path.insert(0,'/home/user/uni/keirin_ai'); sys.path.insert(0,'/home/user/uni/keirin_ai/scripts')
from datetime import datetime
import pandas as pd
import odds_watch as w
from keirin.db import DB
from keirin.sources import kdreams as kd
from keirin.sources.http import PoliteFetcher
from keirin.market_model import market_table, apply_model, one_bet
db=DB('/home/user/uni/keirin_ai/data/keirin.sqlite')
f=PoliteFetcher("keirin-ai-personal/0.1",3.0,'/home/user/uni/keirin_ai/data/cache_kdreams_live')
today=datetime.now().strftime('%Y-%m-%d'); now=datetime.now().strftime('%H:%M')
races=db.df("select race_id,date,venue,race_no,start_time,race_class from races where date=? and start_time>? and race_class not like '%ガールズ%' order by start_time",(today,now)).to_dict('records')
M=pd.read_pickle('/home/user/uni/keirin_ai/data/market_models/odds_only_v1.pkl')
RULES=[('30倍・EV1.1',30,1.1),('30倍・EV1.2',30,1.2),('10倍・EV1.1',10,1.1),('10倍・EV1.2',10,1.2)]
for r in races:
    info=kd.parse_odds(f.get(kd.odds_url(r['race_id'],w.meet_start(db,r['race_id'][8:10],today))))
    if not kd.check_page(info,r): print(r['venue'],r['race_no'],'ページずれ'); continue
    ent=db.df("select car_no,scratched from entries where race_id=?",(r['race_id'],))
    if (ent.scratched==1).any(): print(r['venue'],r['race_no'],'欠車あり→対象外'); continue
    tab=apply_model(market_table(info['odds'],len(ent)),M)
    best={}
    for name,mx,th in RULES:
        t=tab[tab.odds<=mx]; b=t.loc[t.ev.idxmax()]
        best[name]=(b.bt,b.combo,b.odds,round(b.ev,2),b.ev>=th)
    ln=db.df("select car_no,line_id,line_pos from entries where race_id=? order by line_id,line_pos",(r['race_id'],)).groupby('line_id').car_no.apply(lambda s:'-'.join(map(str,s))).tolist()
    print(f"{r['venue']}{r['race_no']}R {r['start_time']} {r['race_class']} 並び{ln} オッズ時点{info['as_of']}")
    for name,(bt,cb,o,ev,ok) in best.items(): print(f"   {name}: {bt} {cb} {o}倍 EV{ev} {'★買い' if ok else '見送り'}")
