"""CSV 取り込み。templates フォルダのCSVと同じ列で作ったファイルを inbox に置くと読み込む。

ファイル名の先頭で種類を判定する:
  races*.csv   … レース情報
  entries*.csv … 出走表（finish 列に着順を入れれば結果も一緒に入る）
  odds*.csv    … オッズ
  payouts*.csv … 払戻金
Excel で作ったCSV（Shift_JIS）でも UTF-8 でも読める。読み込み後は inbox/done に移動。
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

from ..db import DB, now_str

log = logging.getLogger(__name__)


def _read(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, dtype={"race_id": str, "rider_id": str, "combo": str})
        except UnicodeDecodeError:
            continue
    raise ValueError(f"文字コードが読めません: {path}")


def _none(v):
    return None if pd.isna(v) else v


def import_folder(db: DB, folder: str | Path) -> dict[str, int]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / "done"
    done.mkdir(exist_ok=True)
    counts = {"races": 0, "entries": 0, "odds": 0, "payouts": 0}
    files = sorted(folder.glob("*.csv"), key=lambda p: ["races", "entries", "odds", "payouts"].index(
        next((k for k in counts if p.name.startswith(k)), "payouts")))
    for f in files:
        kind = next((k for k in counts if f.name.startswith(k)), None)
        if kind is None:
            log.warning("種類が分からないので飛ばします: %s", f.name)
            continue
        df = _read(f)
        if kind == "races":
            for r in df.to_dict("records"):
                db.upsert_race({k: _none(v) for k, v in r.items()})
        elif kind == "entries":
            rows = [{k: _none(v) for k, v in r.items()} for r in df.to_dict("records")]
            db.upsert_entries(rows)
            for rid in {r["race_id"] for r in rows if r.get("finish") is not None}:
                db.con.execute("UPDATE races SET has_result=1 WHERE race_id=?", (rid,))
        elif kind == "odds":
            if "fetched_at" not in df:
                df["fetched_at"] = now_str()
            if "is_final" not in df:
                df["is_final"] = 0
            for (rid, bt, ts, fin), g in df.groupby(["race_id", "bet_type", "fetched_at", "is_final"]):
                db.insert_odds(rid, bt, dict(zip(g["combo"], g["odds"])), bool(fin), ts)
        elif kind == "payouts":
            for (rid, bt), g in df.groupby(["race_id", "bet_type"]):
                db.upsert_payouts(rid, bt, dict(zip(g["combo"], g["payout"])))
        counts[kind] += len(df)
        db.commit()
        shutil.move(str(f), done / f.name)
        log.info("取り込み: %s (%d行)", f.name, len(df))
    return counts
