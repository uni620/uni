"""今日のレースの「締切前オッズ」を Kドリームスから記録する（発走の約10分前と約3分前の2回）。
3分前のオッズで、1レース1点ルール（keirin/market_model.py）の仮想購入も記録する。

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
from keirin.line_combos import line_combo_probs  # noqa: E402
from keirin.market_model import apply_model, market_table, one_bet  # noqa: E402

import pandas as pd  # noqa: E402

DECIDE_MARK = 3          # この時点（発走3分前）のオッズで仮想購入を決める
LINE_BASE = "LightGBM+競走得点"
# (記録名, モデルファイル, ライン連動を使うか, 期待値の基準, オッズ上限)。ルールは 2026-11-20 に固定（事前登録）。
RULES = [("オッズ1点・EV1.1", "odds_only_v1.pkl", False, 1.1, 30),
         ("オッズ1点・EV1.2", "odds_only_v1.pkl", False, 1.2, 30),
         ("オッズ+ライン1点・EV1.1", "odds_line_v1.pkl", True, 1.1, 30),
         ("オッズ+ライン1点・EV1.2", "odds_line_v1.pkl", True, 1.2, 30),
         ("オッズ1点・10倍以下EV1.1", "odds_only_v1.pkl", False, 1.1, 10)]   # 11/20 追加: 的中が多くブレが小さい
_MODELS: dict = {}


def decide(db: DB, race: dict, odds: dict, ts: str) -> int:
    """締切前オッズで、1レース1点のルールの仮想購入を記録する。欠車があるレースは記録しない（学習と同じ条件）。"""
    ent = db.df("SELECT car_no, scratched, line_id, line_pos, line_size FROM entries WHERE race_id=?", (race["race_id"],))
    if ent.empty or (ent["scratched"] == 1).any():
        return 0
    cars = sorted(ent["car_no"].astype(int).tolist())
    wp = db.method_preds(race["race_id"], LINE_BASE)
    lp = None
    if wp and all(c in wp for c in cars):
        lines = {int(r.car_no): (r.line_id, r.line_pos, r.line_size) for r in ent.itertuples() if pd.notna(r.line_id)}
        lp = line_combo_probs(cars, [wp[c] for c in cars], lines)
    n = 0
    picks: list[str] = []
    for name, mfile, use_line, th, mx in RULES:
        if use_line and lp is None:
            continue
        if mfile not in _MODELS:
            _MODELS[mfile] = pd.read_pickle(ROOT / "data" / "market_models" / mfile)
        tab = market_table(odds, len(cars), lp if use_line else None)
        if tab is None:
            return n
        b = one_bet(apply_model(tab, _MODELS[mfile]), th, mx)
        if b is None:
            continue
        cur = db.con.execute(
            """INSERT OR IGNORE INTO bets (mode, race_id, date, strategy, bet_type, combo, stake, prob, odds,
                   odds_final, ev, decided_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("live", race["race_id"], race["date"], name, b["bt"], b["combo"], 100, float(b["pm"]),
             float(b["odds"]), 0, float(b["ev"]), ts))
        n += cur.rowcount
        picks.append(f"{name}:{b['bt']}{b['combo']} {b['odds']}倍 EV{b['ev']:.2f}")
    db.commit()
    if picks:
        print(f"通知|{race.get('venue')}{race.get('race_no')}R {race.get('start_time')}発走|" + " / ".join(picks), flush=True)
    elif lp is not None or True:
        print(f"通知|{race.get('venue')}{race.get('race_no')}R {race.get('start_time')}発走|見送り（基準以上の目なし）", flush=True)
    return n

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
                        if m == DECIDE_MARK and not info["final"]:
                            nb = decide(db, r, info["odds"], ts)
                            if nb:
                                print("仮想購入", r["race_id"], nb, "点", flush=True)
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
