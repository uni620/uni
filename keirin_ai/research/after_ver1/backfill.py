"""保存済みの出走表ページ（キャッシュ）から、追加の列だけを読み直して書き込む。netkeirin へは取りに行かない。"""
import sys, hashlib, os
sys.path.insert(0, '/home/user/uni/keirin_ai')
from keirin.db import DB, EXTRA_ENTRY_COLS
from keirin.sources.netkeirin import parse_entry
BASE = "https://keirin.netkeiba.com"
db = DB(sys.argv[1]); cache = '/home/user/uni/keirin_ai/data/cache'
rids = [r[0] for r in db.con.execute("SELECT race_id FROM races")]
done = miss = 0
for rid in rids:
    p = f"{cache}/{hashlib.sha1(f'{BASE}/race/entry/?race_id={rid}'.encode()).hexdigest()}.html"
    if not os.path.exists(p):
        miss += 1; continue
    _, rows = parse_entry(open(p, encoding='utf-8').read(), rid)
    for r in rows:
        db.con.execute(f"UPDATE entries SET {','.join(c+'=?' for c in EXTRA_ENTRY_COLS)} WHERE race_id=? AND car_no=?",
                       [r.get(c) for c in EXTRA_ENTRY_COLS] + [rid, r['car_no']])
    done += 1
    if done % 1000 == 0: db.commit()
db.commit()
print("読み直し", done, "レース / キャッシュなし", miss)
print(db.con.execute(f"SELECT COUNT(*), {','.join('COUNT('+c+')' for c in EXTRA_ENTRY_COLS)} FROM entries").fetchone())
