"""今日のレースの「締切前オッズ」を Kドリームスから記録する（発走の約10分前と約3分前の2回）。

- 検証用: 確定オッズで見つけた傾向が、実際に買える時点のオッズでも残るかを確かめる。
- 個人で使うだけ。取ったデータは GitHub などに置かない（data/ は git の対象外）。
- ログインはしない。PoliteFetcher で 3 秒以上あけて取る。キャッシュは使わない（毎回最新）。

使い方: TZ=Asia/Tokyo python scripts/odds_watch.py [最大秒数=7000]
"""
from __future__ import annotations

import sys
import time
from datetime import date as D, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keirin.db import DB  # noqa: E402
from keirin.sources import kdreams as kd  # noqa: E402
from keirin.sources.http import PoliteFetcher  # noqa: E402

MARKS = (10, 3)          # 発走の何分前に取るか
LATE_OK = 1.5            # 予定時刻から何分遅れまでなら取りにいくか（発走前に限る）


def meet_start(db: DB, venue_code: str, d: str) -> str:
    days = {r[0] for r in db.con.execute(
        "SELECT DISTINCT date FROM races WHERE substr(race_id,9,2)=?", (venue_code,))}
    x = D.fromisoformat(d)
    while (x - timedelta(days=1)).isoformat() in days:
        x -= timedelta(days=1)
    return x.isoformat()


def main(limit: float = 7000):
    db = DB(str(ROOT / "data" / "keirin.sqlite"))
    f = PoliteFetcher("keirin-ai-personal/0.1", 3.0, str(ROOT / "data" / "cache_kdreams_live"))
    t0 = time.time()
    today = datetime.now().strftime("%Y-%m-%d")
    done: set[tuple[str, int]] = set()
    ok = bad = 0
    while time.time() - t0 < limit:
        now = datetime.now()
        races = db.df("SELECT race_id,date,venue,race_no,start_time FROM races WHERE date=? AND start_time IS NOT NULL",
                      (today,)).to_dict("records")
        if not races:
            print("今日のレースが DB にない（朝の取り込み待ち）", flush=True)
            time.sleep(300)
            continue
        pending = 0
        for r in races:
            st = datetime.strptime(f"{today} {r['start_time']}", "%Y-%m-%d %H:%M")
            for m in MARKS:
                if (r["race_id"], m) in done:
                    continue
                due = st - timedelta(minutes=m)
                if now > st or now > due + timedelta(minutes=LATE_OK):
                    done.add((r["race_id"], m))        # 間に合わなかった
                    continue
                pending += 1
                if now < due:
                    continue
                try:
                    url = kd.odds_url(r["race_id"], meet_start(db, r["race_id"][8:10], today))
                    if url is None:
                        done.add((r["race_id"], m)); continue
                    html = f.get(url)
                    info = kd.parse_odds(html)
                    if not kd.check_page(info, r) or not info["odds"]:
                        bad += 1; print("ずれ", r["race_id"], info["title"][:40], flush=True)
                    else:
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        for bt, od in info["odds"].items():
                            db.insert_odds(r["race_id"], bt, od, info["final"], ts)
                        db.commit(); ok += 1
                except Exception as e:  # noqa: BLE001
                    bad += 1; print("失敗", r["race_id"], str(e)[:80], flush=True)
                done.add((r["race_id"], m))
                now = datetime.now()
        if pending == 0:
            print("今日の分は終わり", flush=True)
            break
        time.sleep(15)
    print(f"--- 終了 {datetime.now():%H:%M} 成功{ok} 失敗{bad}", flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 7000)
