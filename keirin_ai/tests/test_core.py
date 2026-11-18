from datetime import datetime

import numpy as np
import pytest

from keirin import pipeline
from keirin.combos import combo_probs
from keirin.config import load_config
from keirin.db import DB
from keirin.report import write_report
from keirin.sources.sample import SampleSource
from keirin.strategy import BASELINE, EV, HONMEI, decide_bets


@pytest.fixture
def cfg(tmp_path):
    src = load_config().raw
    raw = {**src, "paths": {"db": "t.sqlite", "excel": "r.xlsx", "model": "m.json", "log": "l.log"}}
    raw["model"] = {**raw["model"], "min_train_races": 50}
    from keirin.config import Config
    return Config(raw=raw, base=tmp_path)


def test_combo_probs_sum_to_one():
    p = np.array([0.3, 0.2, 0.15, 0.12, 0.1, 0.08, 0.05])
    cars = list(range(1, 8))
    out = combo_probs(cars, p)
    for bt in ("2車複", "2車単", "3連複"):
        assert abs(sum(out[bt].values()) - 1) < 1e-9
    assert len(out["2車単"]) == 42 and len(out["2車複"]) == 21 and len(out["3連複"]) == 35
    # ハービル式: 1>2 = 0.3 * 0.2/0.7
    assert out["2車単"]["1>2"] == pytest.approx(0.3 * 0.2 / 0.7)


def test_decide_bets_rules():
    cars = [1, 2, 3]
    p = [0.6, 0.3, 0.1]
    odds = {"2車複": {"1-2": 1.5, "1-3": 8.0, "2-3": 20.0}}
    cfg = {"bet_types": ["2車複"], "stake": 100, "min_ev": 1.2, "min_prob": 0.02, "max_odds": 150,
           "max_bets_per_race": 3}
    bets = decide_bets(cars, p, odds, {"2車複": False}, cfg)
    by = {}
    for b in bets:
        by.setdefault(b["strategy"], []).append(b["combo"])
    assert by[HONMEI] == ["1-2"]           # 一番確率が高い
    assert by[BASELINE] == ["1-2"]         # 一番オッズが低い
    # 1-3 の確率 = 0.6*0.1/0.4 + 0.1*0.6/0.9 ≈ 0.2167 → EV ≈ 1.73
    assert "1-3" in by[EV] and "1-2" not in by[EV]


def test_settle_won_lost_refund(tmp_path):
    db = DB(tmp_path / "s.sqlite")
    db.upsert_race({"race_id": "R1", "date": "2026-01-01", "venue": "X", "race_no": 1})
    db.upsert_entries([{"race_id": "R1", "car_no": c, "rider_id": f"r{c}"} for c in (1, 2, 3, 4)])
    db.set_results("R1", {1: 1, 2: 2, 3: 3}, scratched=[4])
    db.upsert_payouts("R1", "2車複", {"1-2": 450})
    for combo in ("1-2", "1-3", "2-4"):
        db.con.execute("""INSERT INTO bets (mode, race_id, date, strategy, bet_type, combo, stake, decided_at)
                          VALUES ('live','R1','2026-01-01','AI本命','2車複',?,100,'x')""", (combo,))
    db.commit()
    assert pipeline.settle(db) == 3
    res = dict(db.con.execute("SELECT combo, status || ':' || payout FROM bets").fetchall())
    assert res == {"1-2": "won:450", "1-3": "lost:0", "2-4": "refund:100"}


def test_no_lookahead_odds(tmp_path):
    db = DB(tmp_path / "o.sqlite")
    db.insert_odds("R1", "2車複", {"1-2": 3.0}, False, "2026-01-01 10:00:00")
    db.insert_odds("R1", "2車複", {"1-2": 2.0}, True, "2026-01-01 10:40:00")
    od, fin = db.latest_odds("R1", "2車複", decided_before="2026-01-01 10:25:00")
    assert od == {"1-2": 3.0} and fin is False


def test_end_to_end_backtest_and_report(cfg, tmp_path):
    db = DB(cfg.path("db"))
    src = SampleSource(seed=1, n_riders=300)
    days = [f"2026-01-{d:02d}" for d in range(1, 16)]
    for d in days:
        for race in pipeline.ingest_cards(db, src, d):
            pipeline.ingest_odds(db, src, race)
            pipeline.ingest_result(db, src, race)
    res = pipeline.backtest(db, cfg, "2026-01-11", "2026-01-15")
    assert res["days"] == 5 and res["bets"] > 0
    # 判断はレース前のオッズだけで行われている
    assert db.con.execute("SELECT COUNT(*) FROM bets WHERE odds_final=1").fetchone()[0] == 0
    # バックテスト期間の判断に、その日以降の学習データは使っていない（学習は前日まで）
    model = pipeline.train(db, cfg, before_date="2026-01-11")
    assert model is not None and model.trained_races == 10 * 36
    out = write_report(db, cfg.path("excel"), model)
    assert out.exists()


def test_sample_hides_future_results():
    src = SampleSource(seed=2, n_riders=300, now=datetime(2026, 1, 1, 9, 0))
    race = src.fetch_cards("2026-01-01")[0].race
    assert src.fetch_result(race) is None
