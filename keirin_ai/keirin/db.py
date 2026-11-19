"""SQLite データベース。

テーブル
- races    : レース情報
- entries  : 出走表（1行=1選手）。着順は結果取得後に入る
- odds     : オッズ（取得時刻つき）
- payouts  : 払戻金（100円あたり）
- bets     : 仮想購入の記録（一度書いたら判断内容は書き換えない）
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS races (
    race_id     TEXT PRIMARY KEY,   -- 例: 20261118_立川_07
    date        TEXT NOT NULL,      -- YYYY-MM-DD
    venue       TEXT NOT NULL,
    race_no     INTEGER NOT NULL,
    grade       TEXT,               -- 大会グレード GP/G1/G2/G3/F1/F2
    race_name   TEXT,               -- レース名（一般/予選/準決勝/決勝/特選/ドリーム など）
    race_class  TEXT,               -- S級/A級/A級チャレンジ/ガールズ
    start_time  TEXT,               -- HH:MM（分からなければ空）
    n_riders    INTEGER,
    has_result  INTEGER DEFAULT 0,
    bank_len    REAL,               -- バンクの周長（m）
    straight_len REAL               -- 見なし直線（m）
);
CREATE TABLE IF NOT EXISTS entries (
    race_id     TEXT NOT NULL,
    car_no      INTEGER NOT NULL,   -- 車番
    rider_id    TEXT NOT NULL,      -- 選手登録番号（なければ氏名）
    name        TEXT,
    class_rank  TEXT,               -- SS/S1/S2/A1/A2/A3/L1
    score       REAL,               -- 競走得点
    style       TEXT,               -- 逃/両/追
    age         INTEGER,
    home        INTEGER DEFAULT 0,  -- 地元なら1
    line_id     INTEGER,            -- 予想並びのライン番号（同じラインは同じ番号）
    line_pos    INTEGER,            -- ライン内の位置（1=先頭）
    line_size   INTEGER,            -- ラインの人数（単騎=1）
    scratched   INTEGER DEFAULT 0,  -- 欠車なら1
    finish      INTEGER,            -- 着順（失格・落車・途中棄権は 9）
    kimarite    TEXT,               -- このレースでの決まり手（逃げ/捲り/差し/マーク。なければ空）
    km_nige     INTEGER,            -- 出走表に載っている決まり手の回数（レース前の情報）
    km_makuri   INTEGER,
    km_sashi    INTEGER,
    km_mark     INTEGER,
    sb_s REAL, sb_h REAL, sb_b REAL,     -- 出走表のS・H・B回数
    n1 REAL, n2 REAL, n3 REAL, nout REAL, -- 出走表の1着・2着・3着・着外の回数
    win_rate REAL, ren2 REAL, ren3 REAL, -- 出走表の勝率・2連対率・3連対率（%）
    gear REAL,                           -- ギヤ倍数
    PRIMARY KEY (race_id, car_no)
);
CREATE TABLE IF NOT EXISTS odds (
    race_id     TEXT NOT NULL,
    bet_type    TEXT NOT NULL,      -- 2車複/2車単/3連複
    combo       TEXT NOT NULL,      -- 複は小さい順 "1-3", 単は着順通り "3>1"
    odds        REAL NOT NULL,      -- 倍率（12.3倍なら 12.3）
    fetched_at  TEXT NOT NULL,
    is_final    INTEGER DEFAULT 0,  -- 締切後の確定オッズなら1
    PRIMARY KEY (race_id, bet_type, combo, fetched_at)
);
CREATE TABLE IF NOT EXISTS payouts (
    race_id     TEXT NOT NULL,
    bet_type    TEXT NOT NULL,
    combo       TEXT NOT NULL,
    payout      INTEGER NOT NULL,   -- 100円あたりの払戻金
    PRIMARY KEY (race_id, bet_type, combo)
);
CREATE TABLE IF NOT EXISTS bets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    mode        TEXT NOT NULL,      -- live（本番の仮想購入）/ backtest（過去検証）
    race_id     TEXT NOT NULL,
    date        TEXT NOT NULL,
    strategy    TEXT NOT NULL,      -- AI本命 / AI期待値 / 人気1位
    bet_type    TEXT NOT NULL,
    combo       TEXT NOT NULL,
    stake       INTEGER NOT NULL,
    prob        REAL,               -- AIの的中確率
    odds        REAL,               -- 判断に使ったオッズ（なければ空）
    odds_final  INTEGER,            -- 判断に確定オッズを使ったら1（=参考値扱い）
    ev          REAL,
    decided_at  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open',  -- open/won/lost/refund
    payout      INTEGER DEFAULT 0,  -- 実際の払戻額（円）
    settled_at  TEXT,
    UNIQUE (mode, race_id, strategy, bet_type, combo)
);
CREATE TABLE IF NOT EXISTS predictions (
    race_id     TEXT NOT NULL,
    car_no      INTEGER NOT NULL,
    win_prob    REAL NOT NULL,
    model_ver   TEXT,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (race_id, car_no)
);
CREATE INDEX IF NOT EXISTS idx_races_date ON races(date);
CREATE INDEX IF NOT EXISTS idx_entries_rider ON entries(rider_id);
CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status);
"""


EXTRA_ENTRY_COLS = ["sb_s", "sb_h", "sb_b", "n1", "n2", "n3", "nout", "win_rate", "ren2", "ren3", "gear"]


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class DB:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.con = sqlite3.connect(self.path, timeout=60)  # 取得と朝の処理が重なっても待つ
        self.con.execute("PRAGMA foreign_keys = ON")
        self.con.executescript(SCHEMA)
        self._migrate()

    def _migrate(self):
        """古いデータベースに新しい列を足す。"""
        for table, col in ([("races", "race_name TEXT"), ("races", "race_class TEXT"), ("entries", "kimarite TEXT"),
                           ("entries", "km_nige INTEGER"), ("entries", "km_makuri INTEGER"),
                           ("entries", "km_sashi INTEGER"), ("entries", "km_mark INTEGER"),
                           ("races", "bank_len REAL"), ("races", "straight_len REAL")]
                           + [("entries", f"{c} REAL") for c in EXTRA_ENTRY_COLS]):
            try:
                self.con.execute(f"ALTER TABLE {table} ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass

    def close(self):
        self.con.close()

    # ---------- 書き込み ----------
    def upsert_race(self, r: dict):
        self.con.execute(
            """INSERT INTO races (race_id, date, venue, race_no, grade, race_name, race_class, start_time, n_riders,
                                  bank_len, straight_len)
               VALUES (:race_id, :date, :venue, :race_no, :grade, :race_name, :race_class, :start_time, :n_riders,
                       :bank_len, :straight_len)
               ON CONFLICT(race_id) DO UPDATE SET
                 grade=excluded.grade, race_name=excluded.race_name, race_class=excluded.race_class,
                 start_time=excluded.start_time, n_riders=excluded.n_riders,
                 bank_len=COALESCE(excluded.bank_len, bank_len),
                 straight_len=COALESCE(excluded.straight_len, straight_len)""",
            {"grade": None, "race_name": None, "race_class": None, "start_time": None, "n_riders": None,
             "bank_len": None, "straight_len": None, **r},
        )

    def upsert_entries(self, rows: list[dict]):
        cols = ["race_id", "car_no", "rider_id", "name", "class_rank", "score", "style",
                "age", "home", "line_id", "line_pos", "line_size", "scratched", "finish", "kimarite",
                "km_nige", "km_makuri", "km_sashi", "km_mark"] + EXTRA_ENTRY_COLS
        # 欠車（scratched）は結果ページで 1 になる。出走表を取り直しても 0 に戻さない
        sets = [f"{c}=COALESCE(excluded.{c}, {c})" for c in cols[2:] if c != "scratched"]
        sets.append("scratched=MAX(excluded.scratched, scratched)")
        for r in rows:
            d = {c: r.get(c) for c in cols}
            d["home"] = d["home"] or 0
            d["scratched"] = d["scratched"] or 0
            self.con.execute(
                f"""INSERT INTO entries ({",".join(cols)}) VALUES ({",".join(":" + c for c in cols)})
                    ON CONFLICT(race_id, car_no) DO UPDATE SET {",".join(sets)}""",
                d,
            )

    def set_results(self, race_id: str, finishes: dict[int, int], scratched: list[int] | None = None,
                    kimarite: dict[int, str] | None = None):
        for car, fin in finishes.items():
            self.con.execute("UPDATE entries SET finish=? WHERE race_id=? AND car_no=?", (fin, race_id, car))
        for car, k in (kimarite or {}).items():
            self.con.execute("UPDATE entries SET kimarite=? WHERE race_id=? AND car_no=?", (k, race_id, car))
        for car in scratched or []:
            self.con.execute("UPDATE entries SET scratched=1 WHERE race_id=? AND car_no=?", (race_id, car))
        self.con.execute("UPDATE races SET has_result=1 WHERE race_id=?", (race_id,))

    def insert_odds(self, race_id: str, bet_type: str, odds: dict[str, float], is_final: bool,
                    fetched_at: str | None = None):
        ts = fetched_at or now_str()
        self.con.executemany(
            "INSERT OR REPLACE INTO odds VALUES (?,?,?,?,?,?)",
            [(race_id, bet_type, c, float(o), ts, int(is_final)) for c, o in odds.items()],
        )

    def upsert_payouts(self, race_id: str, bet_type: str, payouts: dict[str, int]):
        self.con.executemany(
            "INSERT OR REPLACE INTO payouts VALUES (?,?,?,?)",
            [(race_id, bet_type, c, int(p)) for c, p in payouts.items()],
        )

    def commit(self):
        self.con.commit()

    # ---------- 読み出し ----------
    def df(self, sql: str, params=()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.con, params=params)

    def entries_with_race(self, until_date: str | None = None, dates: list[str] | None = None) -> pd.DataFrame:
        sql = """SELECT e.*, r.date, r.venue, r.race_no, r.grade, r.race_name, r.race_class, r.start_time, r.has_result,
                        r.bank_len, r.straight_len
                 FROM entries e JOIN races r USING (race_id)"""
        params: list = []
        if until_date:
            sql += " WHERE r.date <= ?"
            params.append(until_date)
        elif dates:
            sql += f" WHERE r.date IN ({','.join('?' * len(dates))})"
            params.extend(dates)
        sql += " ORDER BY r.date, r.venue, r.race_no, e.car_no"
        return self.df(sql, params)

    def latest_odds(self, race_id: str, bet_type: str, decided_before: str | None = None) -> tuple[dict[str, float], bool]:
        """判断時点で手に入るオッズのうち最新のもの。(odds, is_final) を返す。"""
        sql = "SELECT fetched_at, is_final FROM odds WHERE race_id=? AND bet_type=?"
        params: list = [race_id, bet_type]
        if decided_before:
            sql += " AND fetched_at <= ?"
            params.append(decided_before)
        sql += " ORDER BY fetched_at DESC LIMIT 1"
        row = self.con.execute(sql, params).fetchone()
        if not row:
            return {}, False
        ts, is_final = row
        rows = self.con.execute(
            "SELECT combo, odds FROM odds WHERE race_id=? AND bet_type=? AND fetched_at=?",
            (race_id, bet_type, ts),
        ).fetchall()
        return {c: o for c, o in rows}, bool(is_final)
