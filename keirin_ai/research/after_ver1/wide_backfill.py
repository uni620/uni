"""保存済みの結果ページからワイドの払戻を読み直して入れる（新しい取得はしない）。"""
import sys, hashlib, os
sys.path.insert(0,'/home/user/uni/keirin_ai')
from keirin.db import DB
from keirin.sources.netkeirin import parse_result, BASE
db=DB(sys.argv[1]); n=0
ids=[r[0] for r in db.con.execute("select race_id from races where has_result=1 and race_id not in (select race_id from payouts where bet_type='ワイド')")]
for rid in ids:
    p=f"/home/user/uni/keirin_ai/data/cache/{hashlib.sha1(f'{BASE}/race/result/?race_id={rid}'.encode()).hexdigest()}.html"
    if not os.path.exists(p): continue
    r=parse_result(open(p,encoding='utf-8').read(),rid)
    if r and "ワイド" in r.payouts: db.upsert_payouts(rid,"ワイド",r.payouts["ワイド"]); n+=1; db.commit()
    if n%500==0: db.commit()
db.commit(); print("ワイド", n, "/", len(ids))
