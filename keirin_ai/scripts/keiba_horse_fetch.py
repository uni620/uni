"""中央競馬に出走した馬ごとに、血統表（5代）と馬のページを netkeiba から取り込む（個人の分析用。3秒間隔、途中から再開できる）。
使い方: python scripts/keiba_horse_fetch.py [制限秒数] [ped|prof]
  ped  = 血統表 db.netkeiba.com/horse/ped/<id>/ → k_horse_ped
  prof = 馬のページ db.netkeiba.com/horse/<id>/ → k_horses
元のページは data/keiba_cache_horse/<ped|prof>/<id>.html.gz に保存。新しく走った馬から順に取る。
"""
import gzip
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keirin.sources.http import PoliteFetcher  # noqa: E402
from keiba.horse import SCHEMA, parse_ped, parse_profile, save_ped, save_profile  # noqa: E402
from keiba.store import connect  # noqa: E402

limit = float(sys.argv[1]) if len(sys.argv) > 1 else 7000
kind = sys.argv[2] if len(sys.argv) > 2 else "ped"
assert kind in ("ped", "prof")
t0 = time.time()
con = connect(str(ROOT / "data" / "keiba.sqlite"))
con.executescript(SCHEMA)
con.execute("""INSERT OR IGNORE INTO k_horse_jobs (horse_id)
               SELECT horse_id FROM k_results WHERE horse_id IS NOT NULL GROUP BY horse_id""")
con.commit()
cache = ROOT / "data" / "keiba_cache_horse" / kind
cache.mkdir(parents=True, exist_ok=True)
f = PoliteFetcher("keiba-ai-personal/0.1", 3.0, str(ROOT / "data" / "keiba_cache_misc"))
done_col, err_col = f"{kind}_done", f"{kind}_error"
todo = con.execute(f"""SELECT j.horse_id FROM k_horse_jobs j
    JOIN (SELECT horse_id, MAX(race_id) last FROM k_results GROUP BY horse_id) r USING (horse_id)
    WHERE j.{done_col}=0 ORDER BY r.last DESC""").fetchall()
n = ok = 0
for (hid,) in todo:
    if time.time() - t0 >= limit:
        break
    try:
        p = cache / f"{hid}.html.gz"
        if p.exists():
            html = gzip.decompress(p.read_bytes()).decode("utf-8")
        else:
            url = f"https://db.netkeiba.com/horse/ped/{hid}/" if kind == "ped" else f"https://db.netkeiba.com/horse/{hid}/"
            html = f.get(url, use_cache=False, encoding="utf-8")
            p.write_bytes(gzip.compress(html.encode("utf-8")))
        if kind == "ped":
            rows = parse_ped(html, hid)
            if not rows:
                raise ValueError("血統表が無い")
            save_ped(con, rows)
        else:
            d = parse_profile(html, hid)
            if not d:
                raise ValueError("プロフィールが無い")
            save_profile(con, d)
        con.execute(f"UPDATE k_horse_jobs SET {done_col}=1, {err_col}=NULL WHERE horse_id=?", (hid,))
        ok += 1
    except Exception as e:  # noqa: BLE001
        con.execute(f"UPDATE k_horse_jobs SET {err_col}=? WHERE horse_id=?", (str(e)[:200], hid))
    n += 1
    if n % 50 == 0:
        con.commit()
    if n % 500 == 0:
        print(f"[{kind}] {n}件 成功{ok} 経過{(time.time()-t0)/60:.0f}分 直近{hid}", flush=True)
con.commit()
rem = con.execute(f"SELECT COUNT(*) FROM k_horse_jobs WHERE {done_col}=0").fetchone()[0]
print(f"--- [{kind}] 終了 今回{n}件 成功{ok} 残り{rem}頭", flush=True)
