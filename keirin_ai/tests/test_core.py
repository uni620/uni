from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from keirin import pipeline
from keirin.combos import combo_probs
from keirin.config import Config, load_config
from keirin.db import DB
from keirin.features import build_features
from keirin.race_grade import kaku, race_level
from keirin.report import write_report
from keirin.sources.sample import SampleSource
from keirin.strategy import BASELINE, decide_baseline, decide_bets


@pytest.fixture
def cfg(tmp_path):
    raw = dict(load_config().raw)
    raw["paths"] = {"db": "t.sqlite", "excel": "r.xlsx", "model": "m.json", "log": "l.log"}
    return Config(raw=raw, base=tmp_path)


# ---------- 確率・買い方 ----------
def test_combo_probs_sum_to_one():
    p = np.array([0.3, 0.2, 0.15, 0.12, 0.1, 0.08, 0.05])
    out = combo_probs(list(range(1, 8)), p)
    for bt in ("2車複", "2車単", "3連複"):
        assert abs(sum(out[bt].values()) - 1) < 1e-9
    assert out["2車単"]["1>2"] == pytest.approx(0.3 * 0.2 / 0.7)


def test_decide_bets_rules():
    odds = {"2車複": {"1-2": 1.5, "1-3": 8.0, "2-3": 20.0}}
    cfg = {"bet_types": ["2車複"], "stake": 100, "min_ev": 1.2, "min_prob": 0.02, "max_odds": 150,
           "max_bets_per_race": 3}
    bets = decide_bets("X", [1, 2, 3], [0.6, 0.3, 0.1], odds, {"2車複": False}, cfg)
    bets += decide_baseline(odds, {"2車複": False}, cfg)
    by = {}
    for b in bets:
        by.setdefault(b["strategy"], []).append(b["combo"])
    assert by["X・本命"] == ["1-2"] and by[BASELINE] == ["1-2"]
    assert "1-3" in by["X・期待値"] and "1-2" not in by["X・期待値"]


# ---------- 格 ----------
def test_race_level_rules():
    assert race_level("準決勝") == 5 and race_level("準々決勝") == 4 and race_level("決勝") == 6
    assert race_level("初日特選") == 5 and race_level("特選") == 3 and race_level("特別優秀") == 4
    assert race_level("一次予選") == 2 and race_level("特一般") == 1 and race_level("スタールビー賞") == 5
    assert race_level("初日特別選抜") == 5 and race_level("特秀") == 4 and race_level("選抜") == 2
    assert race_level("準決") == 5 and race_level("二予A") == 4
    assert race_level("なぞのレース") is None
    assert kaku("準決勝", "G1") == pytest.approx(5.8)
    assert race_level("なぞのレース", {"なぞのレース": 3}) == 3


# ---------- 特徴量（リークしないこと） ----------
def _toy():
    """同じ2人が 1日目 → 2日目 → 4日後 と走る小さなデータ。"""
    rows = []
    sched = [("2025-01-01", "予選", {"a": 1, "b": 2, "c": 3}, {"a": "逃げ", "b": "マーク"}),
             ("2025-01-02", "準決勝", {"a": 2, "b": 1, "c": 3}, {"b": "差し", "a": "逃げ"}),
             ("2025-01-03", "決勝", {"a": None, "b": None, "c": None}, {})]
    for d, name, fin, km in sched:
        rid = f"{d}_X_01"
        for car, (r, pos) in enumerate([("a", 1), ("b", 2), ("c", 1)], start=1):
            rows.append({"race_id": rid, "car_no": car, "rider_id": r, "date": d, "venue": "X", "race_no": 1,
                         "grade": "F1", "race_name": name, "race_class": "S級", "line_id": 1 if r != "c" else 2,
                         "line_pos": pos, "line_size": 2 if r != "c" else 1, "scratched": 0,
                         "finish": fin[r], "kimarite": km.get(r), "score": 100.0})
    return pd.DataFrame(rows)


def test_features_use_only_past():
    f = build_features(_toy()).set_index(["date", "rider_id"])
    # 1日目: 過去が無いので自力度・直近着順は空
    assert np.isnan(f.loc[("2025-01-01", "a"), "jiriki"])
    assert np.isnan(f.loc[("2025-01-01", "a"), "recent_finish"])
    # 2日目: 1日目の結果だけ（a は1着、逃げ1回）
    assert f.loc[("2025-01-02", "a"), "recent_finish"] == 1
    assert f.loc[("2025-01-02", "a"), "jiriki"] == 1.0
    assert f.loc[("2025-01-02", "a"), "is_current_meet"] == 1
    # 3日目: 1・2日目の平均、自力度は 逃げ2回 / 2回、b は マーク1+差し1 → 0
    assert f.loc[("2025-01-03", "a"), "recent_finish"] == 1.5
    assert f.loc[("2025-01-03", "b"), "jiriki"] == 0.0
    # 準決勝の着順は3日目だけに入る（2日目の自分の結果は2日目には使わない）
    assert np.isnan(f.loc[("2025-01-02", "a"), "semi_finish"])
    assert f.loc[("2025-01-03", "a"), "semi_finish"] == 2
    # 位置の適性: 先頭なら自力度、番手なら 1-自力度
    assert f.loc[("2025-01-03", "a"), "pos_fit"] == 1.0
    assert f.loc[("2025-01-03", "b"), "pos_fit"] == 1.0
    # 格の差: 普段（予選2.2, 準決勝5.2 の平均）− 今回（決勝6.2）
    assert f.loc[("2025-01-03", "a"), "kaku_diff"] == pytest.approx((2.2 + 5.2) / 2 - 6.2)


def test_girls_excluded():
    df = _toy()
    df.loc[df["date"] == "2025-01-01", "race_class"] = "ガールズ"
    f = build_features(df)
    assert "2025-01-01" not in set(f["date"])


# ---------- 精算・オッズ ----------
def test_settle_won_lost_refund(tmp_path):
    db = DB(tmp_path / "s.sqlite")
    db.upsert_race({"race_id": "R1", "date": "2026-01-01", "venue": "X", "race_no": 1})
    db.upsert_entries([{"race_id": "R1", "car_no": c, "rider_id": f"r{c}"} for c in (1, 2, 3, 4)])
    db.set_results("R1", {1: 1, 2: 2, 3: 3}, scratched=[4])
    db.upsert_payouts("R1", "2車複", {"1-2": 450})
    for combo in ("1-2", "1-3", "2-4"):
        db.con.execute("""INSERT INTO bets (mode, race_id, date, strategy, bet_type, combo, stake, decided_at)
                          VALUES ('live','R1','2026-01-01','X','2車複',?,100,'x')""", (combo,))
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


# ---------- 全体の流れ ----------
def test_end_to_end_evaluate_and_report(cfg):
    db = DB(cfg.path("db"))
    src = SampleSource(seed=1, n_riders=900, start="2025-01-01")
    for i in range(40):
        d = f"2025-{1 + i // 31:02d}-{1 + i % 31:02d}"
        for race in pipeline.ingest_cards(db, src, d):
            pipeline.ingest_odds(db, src, race)
            pipeline.ingest_result(db, src, race)
    table = pipeline.evaluate_split(db, cfg, test_frac=0.25, out_json=cfg.base / "e.json")
    assert set(table["予想方法"]) == {"手作り7:3", "LightGBM", "LightGBM+競走得点"}
    # 判断はレース前のオッズだけ
    assert db.con.execute("SELECT COUNT(*) FROM bets WHERE odds_final=1").fetchone()[0] == 0
    # 検証期間より前のレースには仮想購入していない
    cut = __import__("json").loads((cfg.base / "e.json").read_text())["cut_date"]
    assert db.con.execute("SELECT COUNT(*) FROM bets WHERE date < ?", (cut,)).fetchone()[0] == 0
    out = write_report(db, cfg.path("excel"), __import__("json").loads((cfg.base / "e.json").read_text()))
    assert out.exists()


def test_sample_hides_future_results():
    src = SampleSource(seed=2, now=datetime(2025, 1, 1, 9, 0), start="2025-01-01")
    race = src.fetch_cards("2025-01-01")[0].race
    assert src.fetch_result(race) is None


# ---------- netkeirin の読み取り（ページの形だけをまねた手書きのHTMLで確認） ----------
from keirin.features import _norm_kimarite  # noqa: E402
from keirin.sources.netkeirin import (apply_narabi, parse_entry, parse_narabi, parse_race_list,  # noqa: E402
                                      parse_result, split_race_name)

ENTRY = """<html><head><title>テスト競輪 テスト杯 GIII 出走表</title></head><body>
<span class="Icon_GradeType">GIII</span><div class="Race_Name">Ｓ級 準決勝</div>
<div class="Race_Data">発走 9:05 締切 9:00</div>
<table id="RaceCard_Table_Fixed"><thead><tr class="Header"><th>枠</th><th>車</th><th></th><th>本紙</th><th>選手名</th>
<th>競走得点</th><th>脚質</th><th>S</th><th>H</th><th>B</th><th>逃げ</th><th>まくり</th><th>差し</th><th>マ｜ク</th></tr></thead>
<tbody>
<tr class="PlayerList"><td>1</td><td>1</td><td></td><td></td><td><a id="names_100_202601010101">選手A</a></td>
<td>110.5</td><td><span>1</span>逃</td><td>1</td><td>0</td><td>0</td><td>4</td><td>2</td><td>0</td><td>0</td></tr>
<tr class="PlayerList"><td>2</td><td>2</td><td></td><td></td><td><a id="names_200_202601010101">選手B</a></td>
<td>105.0</td><td><span>3</span>追</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>3</td></tr>
<tr class="PlayerList"><td>3</td><td>3</td><td></td><td></td><td><a id="names_300_202601010101">選手C</a></td>
<td>100.0</td><td><span>2</span>両</td><td>0</td><td>0</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td></tr>
</tbody></table></body></html>"""

NARABI = """<div class="DeployBox">
<div class="Shaban_InBox"><span class="Waku1 Shaban_Num">1</span></div>
<div class="Shaban_InBox"><span class="Waku2 Shaban_Num">2</span></div>
<div class="Shaban_InBox"><span class="WakuSeparat"></span></div>
<div class="Shaban_InBox"><span class="Waku3 Shaban_Num">3</span></div></div>"""

RESULT = """<table id="All_Result_Table"><tr class="Header"><th>着</th><th>枠番</th><th>車番</th><th>選手名</th>
<th>着差</th><th>上り</th><th>決</th><th>SB</th></tr>
<tr class="PlayerList"><td>1着</td><td>2</td><td>2</td><td>B</td><td></td><td>11.0</td><td>差</td><td></td></tr>
<tr class="PlayerList"><td>2着</td><td>1</td><td>1</td><td>A</td><td></td><td>11.2</td><td>逃</td><td>B</td></tr>
<tr class="PlayerList"><td>失</td><td>3</td><td>3</td><td>C</td><td></td><td></td><td></td><td></td></tr></table>
<table class="Payout_Detail_Table">
<tr class="Umaren"><th>２車複</th><td class="Result">1-2</td><td class="Payout"><span>1,230円</span></td></tr>
<tr class="Umatan"><th>２車単</th><td class="Result">2&gt;1</td><td class="Payout"><span>2,450円</span></td></tr>
<tr class="Fuku3"><th>３連複</th><td class="Result">3-1-2</td><td class="Payout"><span>990円</span></td></tr></table>"""


def test_netkeirin_parsers():
    race, rows = parse_entry(ENTRY, "202601010101")
    assert race["race_class"] == "S級" and race["race_name"] == "準決勝" and race["grade"] == "G3"
    assert race["start_time"] == "09:05" and race["venue"] == "テスト" and race["date"] == "2026-01-01"
    assert [r["rider_id"] for r in rows] == ["100", "200", "300"]
    assert rows[0]["score"] == 110.5 and rows[0]["km_nige"] == 4 and rows[1]["km_mark"] == 3
    lines = parse_narabi(NARABI)
    assert lines == [[1, 2], [3]]
    apply_narabi(rows, lines)
    assert (rows[1]["line_id"], rows[1]["line_pos"], rows[1]["line_size"]) == (1, 2, 2)
    assert rows[2]["line_size"] == 1
    res = parse_result(RESULT, "202601010101")
    assert res.finishes == {2: 1, 1: 2, 3: 9} and res.kimarite == {2: "差", 1: "逃"}
    assert res.payouts == {"2車複": {"1-2": 1230}, "2車単": {"2>1": 2450}, "3連複": {"1-2-3": 990}}


def test_race_name_split_and_kimarite():
    assert split_race_name("Ｌ級ガールズ予選１") == ("ガールズ", "予選1")
    assert split_race_name("Ａ級チャレンジ 一般") == ("A級チャレンジ", "一般")
    assert _norm_kimarite("ク") == "マーク" and _norm_kimarite("捲") == "捲り"
    ids, venues = parse_race_list('<li id="btn_jyoname_28"><a>立川<span class="Icon_Grade">GIII</span></a></li>'
                                  '<a href="x?race_id=202601012801">1R</a>')
    assert ids == ["202601012801"] and venues == {"28": {"venue": "立川", "grade": "G3"}}
