"""過去レースの3連単の確定オッズを Kドリームスから取り込む（新しいレースから順に。途中で止めても続きから）。
締切前オッズの記録（odds_watch.py）が動いている間は取りに行かない（Kドリームスへのアクセスを二重にしないため）。"""
import sys, time, logging, subprocess
from datetime import date as D, timedelta, datetime
sys.path.insert(0, '/home/user/uni/keirin_ai')
from keirin.db import DB
from keirin.sources.http import PoliteFetcher
from keirin.sources import kdreams as kd
logging.basicConfig(level=logging.WARNING)
db = DB('/home/user/uni/keirin_ai/data/keirin.sqlite')
CACHE = '/home/user/uni/keirin_ai/data/cache_kdreams'
f = PoliteFetcher('keirin-ai-personal/0.1', 3.0, CACHE)
limit = float(sys.argv[1]) if len(sys.argv) > 1 else 7000
t0 = time.time()
days = {}
for v, d in db.con.execute("select distinct substr(race_id,9,2), date from races"):
    days.setdefault(v, set()).add(d)
def meet_start(v, d):
    x = D.fromisoformat(d)
    while (x - timedelta(days=1)).isoformat() in days[v]:
        x -= timedelta(days=1)
    return x.isoformat()
def watch_running():
    return subprocess.run(["pgrep", "-f", "odds_watch.py"], capture_output=True).returncode == 0
done = {r[0] for r in db.con.execute("select distinct race_id from odds where is_final=1 and bet_type='3連単'")}
races = db.df("""select race_id,date,venue,race_no from races r where has_result=1 and date>='2026-06-11' and date < date("now","localtime")
                 and exists (select 1 from odds o where o.race_id=r.race_id and o.is_final=1)
                 order by date desc, race_id""").to_dict("records")
todo = [r for r in races if r["race_id"] not in done]
print(f"対象 {len(todo)} / {len(races)}", flush=True)
ok = bad = 0
for i, r in enumerate(todo):
    if time.time() - t0 > limit:
        break
    while watch_running():
        if time.time() - t0 > limit: break
        print("締切前オッズの記録が動いているので待つ", datetime.now().strftime('%H:%M'), flush=True); time.sleep(600)
    url = kd.odds_url(r["race_id"], meet_start(r["race_id"][8:10], r["date"]), "3rentan")
    if url is None:
        bad += 1; continue
    try:
        html = kd.load_section(CACHE, url) or f.get(url)
        info = kd.parse_trifecta(html)
        if not kd.check_page(info, r) or not info["odds"]:
            bad += 1; print("ずれ", r["race_id"], r["venue"], info["title"][:40], flush=True); continue
        kd.save_section(CACHE, url, html)
        ts = info["as_of"].replace("/", "-") + ":00" if info["as_of"] else None
        db.insert_odds(r["race_id"], "3連単", info["odds"]["3連単"], info["final"], ts)
        db.commit(); ok += 1
    except Exception as e:
        bad += 1; print("失敗", r["race_id"], r["venue"], str(e)[:80], flush=True)
    if (i + 1) % 200 == 0:
        print(f"{i+1}件 成功{ok} 失敗{bad} 直近 {r['date']} 経過{(time.time()-t0)/60:.0f}分", flush=True)
print(f"--- 終了 成功{ok} 失敗{bad}", flush=True)
