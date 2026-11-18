"""取得 → 学習 → 予想 → 仮想購入 → 精算 の流れ。"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

from .combos import combo_cars
from .config import Config
from .db import DB, now_str
from .features import build_features
from .model import WinProbModel
from .sources.base import OddsSnapshot, Source
from .strategy import decide_bets

log = logging.getLogger(__name__)


# ---------------- 取得 ----------------
def ingest_cards(db: DB, source: Source, date: str) -> list[dict]:
    cards = source.fetch_cards(date)
    for c in cards:
        db.upsert_race(c.race)
        db.upsert_entries(c.entries)
    db.commit()
    log.info("%s: 出走表 %dレース", date, len(cards))
    return [c.race for c in cards]


def _store_odds(db: DB, snap: OddsSnapshot | None):
    if snap is None:
        return
    for bt, od in snap.odds.items():
        db.insert_odds(snap.race_id, bt, od, snap.is_final, snap.fetched_at)


def ingest_odds(db: DB, source: Source, race: dict):
    _store_odds(db, source.fetch_odds(race))
    db.commit()


def ingest_result(db: DB, source: Source, race: dict) -> bool:
    res = source.fetch_result(race)
    if res is None:
        return False
    db.set_results(res.race_id, res.finishes, res.scratched)
    for bt, p in res.payouts.items():
        db.upsert_payouts(res.race_id, bt, p)
    _store_odds(db, res.final_odds)
    db.commit()
    return True


def pending_result_races(db: DB, until_date: str) -> list[dict]:
    df = db.df("SELECT * FROM races WHERE has_result=0 AND date<=? ORDER BY date, race_no", (until_date,))
    return df.to_dict("records")


# ---------------- 学習 ----------------
def make_model(cfg: Config) -> WinProbModel:
    m = cfg.section("model")
    return WinProbModel(l2=m.get("l2", 1.0), depth=m.get("explode_depth", 3), weights=m.get("explode_weights"))


def features_for(db: DB, cfg: Config, until_date: str | None = None) -> pd.DataFrame:
    raw = db.entries_with_race(until_date=until_date)
    return build_features(raw, cfg.section("model").get("form_window", 10))


def train(db: DB, cfg: Config, feats: pd.DataFrame | None = None, before_date: str | None = None) -> WinProbModel | None:
    if feats is None:
        feats = features_for(db, cfg)
    train_df = feats[feats["finish"].notna()]
    if before_date:
        train_df = train_df[train_df["date"] < before_date]
    n_races = train_df["race_id"].nunique()
    if n_races < cfg.section("model").get("min_train_races", 200):
        log.info("学習データが少ないため学習しません（%dレース）。競走得点ベースの仮予想を使います。", n_races)
        return None
    model = make_model(cfg).fit(train_df)
    return model


# ---------------- 予想・仮想購入 ----------------
def _decision_time(row: dict) -> str:
    if row.get("start_time"):
        dt = datetime.strptime(f"{row['date']} {row['start_time']}", "%Y-%m-%d %H:%M") - timedelta(minutes=5)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return f"{row['date']} 00:00:00"


def predict_and_bet(db: DB, cfg: Config, model: WinProbModel | None, feats: pd.DataFrame, race_ids: list[str],
                    mode: str, decided_at: str | None = None, allow_final_odds: bool = False) -> int:
    """race_ids のレースを予想し、仮想購入を記録する。記録済みのレースは上書きしない。"""
    target = feats[feats["race_id"].isin(race_ids)].copy()
    if target.empty:
        return 0
    m = model if model is not None else WinProbModel()
    target["win_prob"] = m.predict(target)
    scfg = cfg.section("strategy")
    ccfg = cfg.section("combos")
    n = 0
    for rid, g in target.groupby("race_id"):
        g = g.sort_values("car_no")
        row = g.iloc[0].to_dict()
        when = decided_at or _decision_time(row)
        cutoff = None if allow_final_odds else when
        odds_by_type, final_flags = {}, {}
        for bt in scfg.get("bet_types", []):
            od, fin = db.latest_odds(rid, bt, decided_before=cutoff)
            odds_by_type[bt], final_flags[bt] = od, fin
        bets = decide_bets(g["car_no"].astype(int).tolist(), g["win_prob"].to_numpy(), odds_by_type, final_flags,
                           scfg, ccfg.get("lambda2", 1.0), ccfg.get("lambda3", 1.0))
        for car, p in zip(g["car_no"], g["win_prob"]):
            db.con.execute("INSERT OR IGNORE INTO predictions VALUES (?,?,?,?,?)",
                           (rid, int(car), float(p), m.version or "baseline", now_str()))
        for b in bets:
            cur = db.con.execute(
                """INSERT OR IGNORE INTO bets (mode, race_id, date, strategy, bet_type, combo, stake, prob, odds,
                       odds_final, ev, decided_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (mode, rid, row["date"], b["strategy"], b["bet_type"], b["combo"], b["stake"], b["prob"],
                 b["odds"], b["odds_final"], b["ev"], when))
            n += cur.rowcount
    db.commit()
    return n


# ---------------- 精算 ----------------
def settle(db: DB) -> int:
    open_bets = db.df("""SELECT b.id, b.race_id, b.bet_type, b.combo, b.stake FROM bets b
                         JOIN races r USING (race_id) WHERE b.status='open' AND r.has_result=1""")
    if open_bets.empty:
        return 0
    n = 0
    ts = now_str()
    for b in open_bets.to_dict("records"):
        scratched = {r[0] for r in db.con.execute(
            "SELECT car_no FROM entries WHERE race_id=? AND scratched=1", (b["race_id"],))}
        if scratched & set(combo_cars(b["combo"])):
            status, pay = "refund", b["stake"]
        else:
            rows = db.con.execute("SELECT combo, payout FROM payouts WHERE race_id=? AND bet_type=?",
                                  (b["race_id"], b["bet_type"])).fetchall()
            if not rows:
                continue  # 払戻データ待ち
            pays = dict(rows)
            if b["combo"] in pays:
                status, pay = "won", int(round(pays[b["combo"]] * b["stake"] / 100))
            else:
                status, pay = "lost", 0
        db.con.execute("UPDATE bets SET status=?, payout=?, settled_at=? WHERE id=?", (status, pay, ts, b["id"]))
        n += 1
    db.commit()
    return n


# ---------------- 過去検証（バックテスト） ----------------
def backtest(db: DB, cfg: Config, date_from: str, date_to: str, retrain_every: int = 7,
             allow_final_odds: bool = False, progress=None) -> dict:
    """date_from〜date_to の各日について「その日より前のデータだけで学習 → その日を予想」を繰り返す。"""
    db.con.execute("DELETE FROM bets WHERE mode='backtest' AND date BETWEEN ? AND ?", (date_from, date_to))
    db.commit()
    feats = features_for(db, cfg)
    dates = sorted(d for d in feats["date"].unique() if date_from <= d <= date_to)
    model, last_train, total = None, None, 0
    for i, d in enumerate(dates):
        if model is None or last_train is None or i % retrain_every == 0:
            model = train(db, cfg, feats, before_date=d)
            last_train = d
        rids = feats.loc[feats["date"] == d, "race_id"].unique().tolist()
        total += predict_and_bet(db, cfg, model, feats, rids, "backtest", allow_final_odds=allow_final_odds)
        if progress:
            progress(i + 1, len(dates), d)
    settled = settle(db)
    return {"days": len(dates), "bets": total, "settled": settled}
