"""取得 → 学習 → 予想 → 仮想購入 → 精算、と 検証（設計書「検証の方法」）。"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .combos import combo_cars, implied_win_probs_from_exacta
from .config import Config
from .db import DB, now_str
from .features import build_features
from .model import LGBModel, evaluate, make_methods
from .race_grade import load_overrides
from .sources.base import OddsSnapshot, Source
from .strategy import decide_baseline, decide_bets, filtered_bets

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
    db.set_results(res.race_id, res.finishes, res.scratched, res.kimarite)
    for bt, p in res.payouts.items():
        db.upsert_payouts(res.race_id, bt, p)
    _store_odds(db, res.final_odds)
    db.commit()
    return True


def pending_result_races(db: DB, until_date: str) -> list[dict]:
    return db.df("SELECT * FROM races WHERE has_result=0 AND date<=? ORDER BY date, race_no",
                 (until_date,)).to_dict("records")


# ---------------- 特徴量・学習 ----------------
def features_for(db: DB, cfg: Config, until_date: str | None = None) -> pd.DataFrame:
    raw = db.entries_with_race(until_date=until_date)
    f = cfg.section("features")
    overrides = load_overrides(cfg.base / f.get("race_names_csv", "race_names.csv"))
    return build_features(raw, jiriki_threshold=f.get("jiriki_threshold", 0.6),
                          form_days=f.get("form_days", 90), trend_races=f.get("trend_races", 4),
                          kaku_overrides=overrides)


def method_names(cfg: Config) -> list[str]:
    return cfg.section("methods").get("compare", ["手作り7:3", "LightGBM", "LightGBM+競走得点", "LightGBM+競走得点+位置バンク"])


def train_methods(train_df: pd.DataFrame, names: list[str]) -> list:
    out = []
    for m in make_methods(names):
        out.append(m.fit(train_df))
        log.info("学習: %s（%dレース）", m.name, train_df["race_id"].nunique())
    return out


# ---------------- 予想・仮想購入 ----------------
def _decision_time(row: dict) -> str:
    if row.get("start_time"):
        dt = datetime.strptime(f"{row['date']} {row['start_time']}", "%Y-%m-%d %H:%M") - timedelta(minutes=5)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return f"{row['date']} 00:00:00"


def predict_and_bet(db: DB, cfg: Config, methods: list, feats: pd.DataFrame, race_ids: list[str],
                    mode: str, decided_at: str | None = None, allow_final_odds: bool = False,
                    save_predictions_of: str | None = None) -> int:
    """race_ids のレースを各予想方法で予想し、仮想購入を記録する。記録済みの買い目は上書きしない。"""
    target = feats[feats["race_id"].isin(race_ids)].copy()
    if target.empty:
        return 0
    preds = {m.name: m.predict(target) for m in methods}
    scfg, ccfg = cfg.section("strategy"), cfg.section("combos")
    n = 0
    for rid, g in target.groupby("race_id"):
        g = g.sort_values("car_no")
        row = g.iloc[0].to_dict()
        when = decided_at or _decision_time(row)
        cutoff = None if allow_final_odds else when
        odds_by_type, final_flags = {}, {}
        for bt in scfg.get("bet_types", []):
            odds_by_type[bt], final_flags[bt] = db.latest_odds(rid, bt, decided_before=cutoff)
        cars = g["car_no"].astype(int).tolist()
        bets = decide_baseline(odds_by_type, final_flags, scfg)
        for name, p in preds.items():
            wp = p.loc[g.index, "p_win"].to_numpy()
            mb = decide_bets(name, cars, wp, odds_by_type, final_flags, scfg,
                             ccfg.get("lambda2", 1.0), ccfg.get("lambda3", 1.0))
            bets += mb + filtered_bets(mb, row.get("race_class"), scfg)
        if save_predictions_of and save_predictions_of in preds:
            p = preds[save_predictions_of]
            for car, pw in zip(g["car_no"], p.loc[g.index, "p_win"]):
                db.con.execute("INSERT OR IGNORE INTO predictions VALUES (?,?,?,?,?)",
                               (rid, int(car), float(pw), save_predictions_of, now_str()))
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
    n, ts = 0, now_str()
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
            status, pay = ("won", int(round(pays[b["combo"]] * b["stake"] / 100))) if b["combo"] in pays else ("lost", 0)
        db.con.execute("UPDATE bets SET status=?, payout=?, settled_at=? WHERE id=?", (status, pay, ts, b["id"]))
        n += 1
    db.commit()
    return n


# ---------------- 検証（時間で分ける） ----------------
def popularity_pred(db: DB, df: pd.DataFrame, allow_final_odds: bool) -> pd.DataFrame | None:
    """人気順: 2車単オッズから逆算した1着確率。オッズが無いレースは空。"""
    out = pd.Series(np.nan, index=df.index)
    for rid, g in df.groupby("race_id"):
        g = g.sort_values("car_no")
        cutoff = None if allow_final_odds else _decision_time(g.iloc[0].to_dict())
        od, _ = db.latest_odds(rid, "2車単", decided_before=cutoff)
        p = implied_win_probs_from_exacta(g["car_no"].astype(int).tolist(), od)
        if p is not None:
            out.loc[g.index] = p
    if out.notna().sum() == 0:
        return None
    return pd.DataFrame({"p_win": out})


def evaluate_split(db: DB, cfg: Config, test_frac: float = 0.22, allow_final_odds: bool = False,
                   out_json: Path | None = None) -> pd.DataFrame:
    """古いレースで学習し、新しいレースで試す。成績表を返し、仮想購入（backtest）も記録する。"""
    feats = features_for(db, cfg)
    feats = feats[feats["finish"].notna()]
    races = feats[["race_id", "date"]].drop_duplicates().sort_values(["date", "race_id"])
    if len(races) < 50:
        raise ValueError(f"結果のあるレースが {len(races)} しかありません")
    cut_date = races["date"].iloc[int(len(races) * (1 - test_frac))]
    train, test = feats[feats["date"] < cut_date], feats[feats["date"] >= cut_date]
    log.info("学習 %dレース（〜%s） / 検証 %dレース（%s〜）",
             train["race_id"].nunique(), cut_date, test["race_id"].nunique(), cut_date)

    methods = train_methods(train, method_names(cfg))
    rows = []
    pop = popularity_pred(db, test, allow_final_odds) if cfg.section("strategy").get("baseline_enabled", False) else None
    if pop is not None:
        rows.append({"予想方法": "人気順", **evaluate(test, pop.dropna())})
    for m in methods:
        rows.append({"予想方法": m.name, **evaluate(test, m.predict(test))})
    table = pd.DataFrame(rows)

    db.con.execute("DELETE FROM bets WHERE mode='backtest'")
    # 検証期間のレースの予想だけ消す（本番＝今日以降の予想は残す）
    test_ids = test["race_id"].unique().tolist()
    db.con.executemany("DELETE FROM predictions WHERE race_id=?", [(r,) for r in test_ids])
    db.commit()
    main = next((m.name for m in methods if isinstance(m, LGBModel) and not m.use_score), None)
    n = predict_and_bet(db, cfg, methods, feats, test["race_id"].unique().tolist(), "backtest",
                        allow_final_odds=allow_final_odds, save_predictions_of=main)
    settle(db)
    log.info("検証の仮想購入 %d 点", n)
    if out_json:
        info = {"cut_date": cut_date, "train_races": int(train["race_id"].nunique()),
                "test_races": int(test["race_id"].nunique()), "table": table.to_dict("records"),
                "importance": {m.name: m.importance().to_dict("records") for m in methods}}
        Path(out_json).write_text(json.dumps(info, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return table
