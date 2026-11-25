"""中央競馬の過去レースを netkeiba から取り込む（個人の分析用。3秒間隔、robots.txt を守る、途中から再開できる）。

新しい月から順にさかのぼり、目標のレース数（既定 24,599 = 競輪と同じ数）に届いたら止まる。
1) race.netkeiba.com のカレンダー（月ごと）→ 開催日
2) db.netkeiba.com/race/list/<日付>/ → 中央競馬のレースID（場コード01〜10）
3) db.netkeiba.com/race/<ID>/ → レース結果を解析して data/keiba.sqlite に保存。元のページは data/keiba_cache/ に gzip で保存
使い方: python scripts/keiba_fetch.py [制限秒数] [目標レース数]
"""
import gzip
import logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keirin.sources.http import PoliteFetcher  # noqa: E402
from keiba.parse import calendar_dates, list_race_ids, parse_race  # noqa: E402
from keiba.store import connect, save_race  # noqa: E402

limit = float(sys.argv[1]) if len(sys.argv) > 1 else 7000
target = int(sys.argv[2]) if len(sys.argv) > 2 else 24599
t0 = time.time()
con = connect(str(ROOT / "data" / "keiba.sqlite"))
cache = ROOT / "data" / "keiba_cache"
cache.mkdir(parents=True, exist_ok=True)
f = PoliteFetcher("keiba-ai-personal/0.1", 3.0, str(ROOT / "data" / "keiba_cache_misc"))
today = date.today()


def left() -> bool:
    return time.time() - t0 < limit


def n_ids() -> int:
    return con.execute("SELECT COUNT(*) FROM k_race_ids").fetchone()[0]


def get(url: str) -> str:
    return f.get(url, use_cache=False, encoding="utf-8")


# 1+2) 月をさかのぼって、目標数のレースIDが揃うまで集める（取得済みの日はとばす）
y, m = today.year, today.month
while left() and n_ids() < target:
    ym = f"{y}{m:02d}"
    done_month = con.execute("SELECT 1 FROM k_dates WHERE date LIKE ? AND listed=1 LIMIT 1", (ym + "%",)).fetchone()
    if not con.execute("SELECT 1 FROM k_dates WHERE date LIKE ? LIMIT 1", (ym + "%",)).fetchone():
        for d in calendar_dates(get(f"https://race.netkeiba.com/top/calendar.html?year={y}&month={m}")):
            if d <= today.strftime("%Y%m%d"):
                con.execute("INSERT OR IGNORE INTO k_dates (date) VALUES (?)", (d,))
        con.commit()
    for (d,) in con.execute("SELECT date FROM k_dates WHERE date LIKE ? AND listed=0 ORDER BY date DESC", (ym + "%",)).fetchall():
        if not left() or n_ids() >= target:
            break
        if d >= today.strftime("%Y%m%d"):
            continue                                  # 今日以降は結果がまだない
        ids = list_race_ids(get(f"https://db.netkeiba.com/race/list/{d}/"))
        con.executemany("INSERT OR IGNORE INTO k_race_ids (race_id, date) VALUES (?,?)", [(i, d) for i in ids])
        con.execute("UPDATE k_dates SET listed=1 WHERE date=?", (d,))
        con.commit()
    m -= 1
    if m == 0:
        y, m = y - 1, 12
    if y < 2000:
        break
print(f"レースID {n_ids()} 件（目標 {target}）", flush=True)

# 3) 各レースの結果ページ（新しい順）
todo = con.execute("SELECT race_id FROM k_race_ids WHERE fetched=0 ORDER BY race_id DESC").fetchall()
n = ok = 0
for (rid,) in todo:
    if not left():
        break
    try:
        p = cache / f"{rid}.html.gz"
        if p.exists():
            html = gzip.decompress(p.read_bytes()).decode("utf-8")
        else:
            html = get(f"https://db.netkeiba.com/race/{rid}/")
            p.write_bytes(gzip.compress(html.encode("utf-8")))
        d = parse_race(html, rid)
        if not d["results"]:
            raise ValueError("結果の表が無い")
        save_race(con, d)
        con.execute("UPDATE k_race_ids SET fetched=1, error=NULL WHERE race_id=?", (rid,))
        ok += 1
    except Exception as e:  # noqa: BLE001
        con.execute("UPDATE k_race_ids SET error=? WHERE race_id=?", (str(e)[:200], rid))
    n += 1
    if n % 50 == 0:
        con.commit()
    if n % 500 == 0:
        print(f"{n}件 成功{ok} 経過{(time.time()-t0)/60:.0f}分 直近{rid}", flush=True)
con.commit()
rem = con.execute("SELECT COUNT(*) FROM k_race_ids WHERE fetched=0").fetchone()[0]
print(f"--- 終了 今回{n}件 成功{ok} 残り{rem}件 保存済み{con.execute('SELECT COUNT(*) FROM k_races').fetchone()[0]}レース", flush=True)
