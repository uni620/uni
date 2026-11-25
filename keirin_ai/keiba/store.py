"""中央競馬のデータを SQLite（data/keiba.sqlite、git に入れない）に保存する。"""
from __future__ import annotations

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS k_dates (date TEXT PRIMARY KEY, listed INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS k_race_ids (race_id TEXT PRIMARY KEY, date TEXT, fetched INTEGER DEFAULT 0, error TEXT);
CREATE TABLE IF NOT EXISTS k_races (
  race_id TEXT PRIMARY KEY, date TEXT, venue_code TEXT, venue TEXT, kai INTEGER, nichi INTEGER, race_no INTEGER,
  race_name TEXT, surface TEXT, direction TEXT, distance INTEGER, weather TEXT, going TEXT, start_time TEXT,
  class_text TEXT, condition_text TEXT, detail_text TEXT, lap_text TEXT, pace_text TEXT, n_horses INTEGER);
CREATE TABLE IF NOT EXISTS k_results (
  race_id TEXT, umaban TEXT, finish_text TEXT, waku TEXT, horse_name TEXT, horse_id TEXT, sex TEXT, age TEXT,
  weight_carried TEXT, jockey TEXT, jockey_id TEXT, time TEXT, margin TEXT, passing TEXT, last3f TEXT,
  win_odds TEXT, popularity TEXT, body_weight INTEGER, body_weight_diff TEXT, body_weight_text TEXT, remarks TEXT,
  trainer TEXT, trainer_id TEXT, owner TEXT, owner_id TEXT, prize TEXT, all_columns TEXT,
  PRIMARY KEY (race_id, umaban, horse_name));
CREATE TABLE IF NOT EXISTS k_payouts (race_id TEXT, bet_type TEXT, combo TEXT, payout INTEGER, popularity INTEGER,
  PRIMARY KEY (race_id, bet_type, combo));
CREATE TABLE IF NOT EXISTS k_corners (race_id TEXT, corner TEXT, ord TEXT, PRIMARY KEY (race_id, corner));
"""


def connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=60)
    con.executescript(SCHEMA)
    return con


def save_race(con: sqlite3.Connection, d: dict) -> None:
    r = d["race"]
    cols = ["race_id", "date", "venue_code", "venue", "kai", "nichi", "race_no", "race_name", "surface", "direction",
            "distance", "weather", "going", "start_time", "class_text", "condition_text", "detail_text", "lap_text",
            "pace_text", "n_horses"]
    con.execute(f"INSERT OR REPLACE INTO k_races ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                [r.get(c) for c in cols])
    rc = ["race_id", "umaban", "finish_text", "waku", "horse_name", "horse_id", "sex", "age", "weight_carried", "jockey",
          "jockey_id", "time", "margin", "passing", "last3f", "win_odds", "popularity", "body_weight", "body_weight_diff",
          "body_weight_text", "remarks", "trainer", "trainer_id", "owner", "owner_id", "prize", "all_columns"]
    con.executemany(f"INSERT OR REPLACE INTO k_results ({','.join(rc)}) VALUES ({','.join('?' * len(rc))})",
                    [[x.get(c) for c in rc] for x in d["results"]])
    con.executemany("INSERT OR REPLACE INTO k_payouts VALUES (?,?,?,?,?)",
                    [(p["race_id"], p["bet_type"], p["combo"], p["payout"], p["popularity"]) for p in d["payouts"]])
    con.executemany("INSERT OR REPLACE INTO k_corners VALUES (?,?,?)",
                    [(c["race_id"], c["corner"], c["order"]) for c in d["corners"]])
