"""今日のレースの「締切前オッズ」を Kドリームスから記録する（発走の約10分前と約3分前の2回）。
8分前のオッズで、1レース1点ルール（keirin/market_model.py）の仮想購入も記録する。

- 検証用: 確定オッズで見つけた傾向が、実際に買える時点のオッズでも残るかを確かめる。
- 個人で使うだけ。取ったデータは GitHub などに置かない（data/ は git の対象外）。
- ログインはしない。PoliteFetcher で 3 秒以上あけて取る。キャッシュは使わない（毎回最新）。

使い方: TZ=Asia/Tokyo python scripts/odds_watch.py [最大秒数=7000]
"""
from __future__ import annotations

import os
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
from keirin.ana import ANA_NAME, GYOKUSAI_NAME, ana_pick, gyokusai_bets  # noqa: E402
from keirin.head import BET_3T, BET_3T_EV, EV_MIN  # noqa: E402

import pandas as pd  # noqa: E402

DECIDE_MARK = 8          # この時点（発走8分前）のオッズで仮想購入を決める（締切に間に合うように）
LINE_BASE = "LightGBM+競走得点"
# (記録名, モデルファイル, ライン連動を使うか, 期待値の基準, オッズ上限)。ルールは 2026-11-20 に固定（事前登録）。
RULES = [("オッズ1点・EV1.1", "odds_only_v1.pkl", False, 1.1, 30),
         ("オッズ1点・EV1.2", "odds_only_v1.pkl", False, 1.2, 30),
         ("オッズ+ライン1点・EV1.1", "odds_line_v1.pkl", True, 1.1, 30),
         ("オッズ+ライン1点・EV1.2", "odds_line_v1.pkl", True, 1.2, 30),
         ("オッズ1点・10倍以下EV1.1", "odds_only_v1.pkl", False, 1.1, 10),   # 11/20 追加: 的中が多くブレが小さい
         ("ver1（オッズ1点・10倍以下EV1.2）", "odds_only_v1.pkl", False, 1.2, 10),   # 11/22 ver1 として固定（検証で一番高かった）
         ("ver1b（ワイドなし・10倍以下EV1.2）", "odds_only_v1.pkl", False, 1.2, 10),  # 11/22 ユーザー指示: ワイドを買わない版
         ("ver1c（ワイドなし・確定見込みEV1.2）", "odds_only_v1.pkl", False, 1.2, 10)]  # 11/22: 確定までの下がり方を見込む（通知はこれ）
NO_WIDE = {"ver1b（ワイドなし・10倍以下EV1.2）", "ver1c（ワイドなし・確定見込みEV1.2）"}
# 8分前→確定のオッズの下がり方（実際に買った目の中央値、11/20〜22の64レース。2車単・3連複は2車複と同じと仮定）。データが増えたら見直す
DRIFT = {"2車複": 0.79, "2車単": 0.79, "3連複": 0.79, "ワイド": 0.68}
DRIFT_RULES = {"ver1c（ワイドなし・確定見込みEV1.2）"}
_MODELS: dict = {}


def decide(db: DB, race: dict, odds: dict, ts: str) -> int:
    """締切前オッズで、1レース1点のルールの仮想購入を記録する。欠車があるレースは記録しない（学習と同じ条件）。
    ガールズは記録しない（ver1 などのルールはガールズ以外のレースで決めたため。オッズだけは記録する）。"""
    if "ガールズ" in str(race.get("race_class") or db.con.execute("SELECT race_class FROM races WHERE race_id=?", (race["race_id"],)).fetchone()[0]):
        return 0
    ent = db.df("SELECT car_no, scratched, line_id, line_pos, line_size FROM entries WHERE race_id=?", (race["race_id"],))
    if ent.empty or (ent["scratched"] == 1).any():
        return 0
    cars = sorted(ent["car_no"].astype(int).tolist())
    wp = db.method_preds(race["race_id"], LINE_BASE)
    # 通知用の予想: ライン連動の本命（無ければ LightGBM の保存済み予想で作る）
    wp_view = wp or {int(c): float(p) for c, p in db.con.execute(
        "SELECT car_no, win_prob FROM predictions WHERE race_id=?", (race["race_id"],))}
    yoso = ""
    if wp_view and all(c in wp_view for c in cars):
        lines_v = {int(r.car_no): (r.line_id, r.line_pos, r.line_size) for r in ent.itertuples() if pd.notna(r.line_id)}
        pv = [wp_view[c] for c in cars]
        lpv = line_combo_probs(cars, pv, lines_v)
        top = sorted(zip(cars, pv), key=lambda x: -x[1])[:3]
        yoso = ("予想 " + " ".join(f"{c}番{p*100:.0f}%" for c, p in top) + " 本命 "
                + " ".join(f"{bt}{max(lpv[bt], key=lpv[bt].get)}" for bt in ("2車複", "3連複")))
    lp = None
    if wp and all(c in wp for c in cars):
        lines = {int(r.car_no): (r.line_id, r.line_pos, r.line_size) for r in ent.itertuples() if pd.notna(r.line_id)}
        lp = line_combo_probs(cars, [wp[c] for c in cars], lines)
    n = 0
    picks: list[str] = []
    near = ""
    for name, mfile, use_line, th, mx in RULES:
        if use_line and lp is None:
            continue
        if db.con.execute("SELECT 1 FROM bets WHERE mode='live' AND race_id=? AND strategy=?",
                          (race["race_id"], name)).fetchone():
            continue                                   # 再起動などで同じレースを2回判定しない
        if mfile not in _MODELS:
            _MODELS[mfile] = pd.read_pickle(ROOT / "data" / "market_models" / mfile)
        tab = market_table(odds, len(cars), lp if use_line else None)
        if tab is None:
            return n
        scored = apply_model(tab, _MODELS[mfile])
        if name in NO_WIDE:
            scored = scored[scored["bt"] != "ワイド"]
        if name in DRIFT_RULES:                  # 確定時のオッズを見込んだ期待値（確率×オッズ×下がり方）
            scored = scored.copy()
            f = scored["bt"].map(DRIFT).fillna(0.79)
            scored["pm"] = scored["pm"] * f      # ボーダー = 基準 / pm がそのまま「8分前のオッズで何倍以上なら買い」になる
            scored["ev"] = scored["pm"] * scored["odds"]
        b = one_bet(scored, th, mx)
        if name.startswith("ver1"):              # ver1・ver1b の結果は買い・見送りとも1行で残す（通知用）
            c1 = one_bet(scored, -1, mx)
            if c1 is not None:
                print(f"{name.split('（')[0]}|{race.get('venue')}{race.get('race_no')}R {race.get('start_time')}発走|"
                      f"{'買い' if b is not None else 'なし'}|{c1['bt']}{c1['combo']}|{c1['odds']}|{c1['ev']:.2f}|{c1['pm']:.5f}",
                      flush=True)
        if b is None:
            if not use_line and mx == 30 and th == 1.2:
                c = one_bet(scored, -1, mx)
                if c is not None:
                    near = f"候補 {c['bt']}{c['combo']} {c['odds']}倍 EV{c['ev']:.2f} ボーダー{1.2 / c['pm']:.1f}倍"
            continue
        cur = db.con.execute(
            """INSERT OR IGNORE INTO bets (mode, race_id, date, strategy, bet_type, combo, stake, prob, odds,
                   odds_final, ev, decided_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("live", race["race_id"], race["date"], name, b["bt"], b["combo"], 100, float(b["pm"]),
             float(b["odds"]), 0, float(b["ev"]), ts))
        n += cur.rowcount
        # ボーダー: 当たる確率をそのままとして、期待値が基準になる最低オッズ（これ未満なら見送り）
        picks.append(f"{name}:{b['bt']}{b['combo']} {b['odds']}倍 EV{b['ev']:.2f} ボーダー{th / b['pm']:.1f}倍")
    db.commit()
    if picks:
        print(f"通知|{race.get('venue')}{race.get('race_no')}R {race.get('start_time')}発走|買い " + " / ".join(picks) + f"|{yoso}", flush=True)
    else:
        print(f"通知|{race.get('venue')}{race.get('race_no')}R {race.get('start_time')}発走|見送り {near}|{yoso}", flush=True)
    return n

def decide_ana(db: DB, race: dict, tri: dict, exacta: dict | None, ts: str) -> int:
    """穴の系統（keirin/ana.py）の仮想購入。8分前の3連単（と2車単）のオッズで決める。欠車レースとガールズは記録しない。"""
    if "ガールズ" in str(db.con.execute("SELECT race_class FROM races WHERE race_id=?", (race["race_id"],)).fetchone()[0]):
        return 0
    ent = db.df("SELECT car_no, rider_id, score, line_id, line_pos, line_size, scratched FROM entries WHERE race_id=?",
                (race["race_id"],))
    if ent.empty or (ent["scratched"] == 1).any():
        return 0
    n = 0
    picks = []
    if "ana" not in _MODELS:
        _MODELS["ana"] = pd.read_pickle(ROOT / "data" / "market_models" / "ana_v1.pkl")
    seen = {r[0] for r in db.con.execute("SELECT strategy FROM bets WHERE mode='live' AND race_id=?", (race["race_id"],))}
    if ANA_NAME not in seen:
        b = ana_pick(tri, ent, _MODELS["ana"])
        if b:
            n += db.con.execute("""INSERT OR IGNORE INTO bets (mode, race_id, date, strategy, bet_type, combo, stake, prob, odds,
                   odds_final, ev, decided_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                   ("live", race["race_id"], race["date"], ANA_NAME, "3連単", b["combo"], 100, b["pm"], b["odds"], 0, b["ev"], ts)).rowcount
            picks.append(f"穴v1:3連単{b['combo']} {b['odds']}倍 EV{b['ev']:.2f}")
    if GYOKUSAI_NAME not in seen and exacta:
        for cb in gyokusai_bets(db.con, race["race_id"], race["date"], ent, exacta):
            n += db.con.execute("""INSERT OR IGNORE INTO bets (mode, race_id, date, strategy, bet_type, combo, stake, prob, odds,
                   odds_final, ev, decided_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                   ("live", race["race_id"], race["date"], GYOKUSAI_NAME, "3連単", cb, 100, None, tri.get(cb), 0, None, ts)).rowcount
        if n and GYOKUSAI_NAME not in seen:
            picks.append("玉砕番手v1:12点")
    db.commit()
    if picks:
        print(f"穴の系統|{race.get('venue')}{race.get('race_no')}R|" + " / ".join(picks), flush=True)
    return n


def decide_head(db: DB, race: dict, tri: dict, ts: str) -> int:
    """頭AI v1＋相手AI展開型v1: 朝に記録した3連単9点のうち、見込み確率×8分前オッズ が EV_MIN を超える目だけを記録する。"""
    if db.con.execute("SELECT 1 FROM bets WHERE mode='live' AND race_id=? AND strategy=?", (race["race_id"], BET_3T_EV)).fetchone():
        return 0
    rows = db.con.execute("SELECT combo, prob FROM bets WHERE mode='live' AND race_id=? AND strategy=?",
                          (race["race_id"], BET_3T)).fetchall()
    n, keep = 0, []
    for combo, prob in rows:
        o = tri.get(combo)
        if not o or prob is None or prob * o <= EV_MIN:
            continue
        n += db.con.execute("""INSERT OR IGNORE INTO bets (mode, race_id, date, strategy, bet_type, combo, stake, prob, odds,
               odds_final, ev, decided_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
               ("live", race["race_id"], race["date"], BET_3T_EV, "3連単", combo, 100, prob, o, 0, prob * o, ts)).rowcount
        keep.append(f"{combo}@{o}")
    db.commit()
    if rows:
        print(f"頭AI|{race.get('venue')}{race.get('race_no')}R {race.get('start_time')}発走|"
              f"{'買い ' + ' '.join(keep) if keep else 'EV1.2超なし'}", flush=True)
    return n


def backfill_one(db: DB, f) -> bool:
    """空き時間に、過去レースの3連単の確定オッズを1レース分だけ取る（新しい順、6/11以降、まだ無いもの）。"""
    r = db.con.execute("""SELECT race_id, date, venue, race_no FROM races r WHERE has_result=1 AND date>='2026-06-11'
            AND date < date('now','localtime')
            AND EXISTS (SELECT 1 FROM odds o WHERE o.race_id=r.race_id AND o.is_final=1)
            AND NOT EXISTS (SELECT 1 FROM odds o WHERE o.race_id=r.race_id AND o.bet_type='3連単' AND o.is_final=1)
            AND race_id NOT IN (SELECT race_id FROM bf_skip)
            ORDER BY date DESC LIMIT 1""").fetchone()
    if r is None:
        return False
    rid, d, venue, rno = r
    race = {"race_id": rid, "date": d, "venue": venue, "race_no": rno}
    try:
        url = kd.odds_url(rid, meet_start(db, rid[8:10], d), "3rentan")
        info = kd.parse_trifecta(f.get(url)) if url else None
        if info and kd.check_page(info, race) and info["odds"]:
            ts = info["as_of"].replace("/", "-") + ":00" if info["as_of"] else None
            db.insert_odds(rid, "3連単", info["odds"]["3連単"], info["final"], ts)
        else:
            db.con.execute("INSERT OR IGNORE INTO bf_skip VALUES (?)", (rid,))
    except Exception:  # noqa: BLE001
        db.con.execute("INSERT OR IGNORE INTO bf_skip VALUES (?)", (rid,))
    db.commit()
    return True


MARKS = (8,)             # 発走の8分前だけ取る（2026-11-22 閲覧回数を減らすため 10分前・3分前はやめた）
TRI_MARKS = (8,)         # 3連単も8分前だけ
LATE_OK = 1.5            # 予定時刻から何分遅れまでなら取りにいくか（発走前に限る）


def meet_start(db: DB, venue_code: str, d: str) -> str:
    days = {r[0] for r in db.con.execute(
        "SELECT DISTINCT date FROM races WHERE substr(race_id,9,2)=?", (venue_code,))}
    x = D.fromisoformat(d)
    while (x - timedelta(days=1)).isoformat() in days:
        x -= timedelta(days=1)
    return x.isoformat()


BACKFILL = True


def _next_due_gap(races: list[dict], done: set, today: str) -> float:
    """次の記録予定まで何秒あるか（予定が無ければ大きな値）。"""
    now = datetime.now()
    gaps = [((datetime.strptime(f"{today} {r['start_time']}", "%Y-%m-%d %H:%M") - timedelta(minutes=m)) - now).total_seconds()
            for r in races for m in MARKS if (r["race_id"], m) not in done]
    gaps = [g for g in gaps if g > -LATE_OK * 60]
    return min(gaps) if gaps else 1e9


def main(limit: float = 7000):
    db = DB(str(ROOT / "data" / "keirin.sqlite"))
    db.con.execute("CREATE TABLE IF NOT EXISTS bf_skip (race_id TEXT PRIMARY KEY)")
    f = PoliteFetcher("keirin-ai-personal/0.1", float(os.environ.get("KD_INTERVAL", "3.0")), str(ROOT / "data" / "cache_kdreams_live"))
    t0 = time.time()
    today = datetime.now().strftime("%Y-%m-%d")
    done: set[tuple[str, int]] = set()
    ok = bad = 0
    while time.time() - t0 < limit:
        db.commit()                              # 開いたままの読み取りを閉じ、朝に入った出走表が見えるようにする
        now = datetime.now()
        races = db.df("SELECT race_id,date,venue,race_no,start_time FROM races WHERE date=? AND start_time IS NOT NULL",
                      (today,)).to_dict("records")
        if not races:
            print("今日のレースが DB にない（朝の取り込み待ち）", flush=True)
            t_end = time.time() + 300          # 待つ間も過去の3連単を取る
            while BACKFILL and time.time() < t_end and backfill_one(db, f):
                pass
            time.sleep(max(0.0, t_end - time.time()))
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
                        if m in TRI_MARKS:            # 3連単は別ページ（1回ぶん通信が増える）
                            try:
                                tu = kd.odds_url(r["race_id"], meet_start(db, r["race_id"][8:10], today), "3rentan")
                                ti = kd.parse_trifecta(f.get(tu)) if tu else None
                                if ti and kd.check_page(ti, r) and ti["odds"]:
                                    tts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    db.insert_odds(r["race_id"], "3連単", ti["odds"]["3連単"], ti["final"], tts)
                                    db.commit()
                                    if m == DECIDE_MARK and not ti["final"]:
                                        decide_ana(db, r, ti["odds"]["3連単"], info["odds"].get("2車単"), tts)
                                        decide_head(db, r, ti["odds"]["3連単"], tts)
                            except Exception as e:  # noqa: BLE001
                                print("3連単失敗", r["race_id"], str(e)[:80], flush=True)
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
        # 次に取る予定まで余裕があれば、その間に過去の3連単を取る（同じ取得器なので3秒以上の間隔は守られる）
        t_end = time.time() + 15
        while BACKFILL and time.time() < t_end - 4 and _next_due_gap(races, done, today) > 60:
            if not backfill_one(db, f):
                break
        time.sleep(max(0.0, t_end - time.time()))
    print(f"--- 終了 {datetime.now():%H:%M} 成功{ok} 失敗{bad}", flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 7000)
