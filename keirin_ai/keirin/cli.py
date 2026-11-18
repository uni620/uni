"""コマンド一覧（python -m keirin <コマンド>）

  demo        架空データで「特徴量→学習→検証→Excel」まで一通り試す（ネット不要）
  evaluate    手元のデータで検証（古いレースで学習、新しいレースで試す。設計書「検証の方法」）
  import-csv  inbox フォルダのCSVを取り込む
  fetch       データ取得元から過去のレースをまとめて取り込む（--from / --to）
  predict     今日の発走前レースを予想して仮想購入（本番用）
  daily       結果取得→精算→AI再学習→Excel更新（本番用）
  report      Excelだけ作り直す
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from . import pipeline
from .config import ROOT, load_config
from .db import DB, now_str
from .race_grade import load_overrides, race_level
from .report import summary_table, write_report
from .sources import make_source
from .sources.csv_source import import_folder

log = logging.getLogger("keirin")


def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.handlers = [fh, sh]


def unknown_race_names(db: DB, cfg) -> pd.DataFrame:
    over = load_overrides(cfg.base / cfg.section("features").get("race_names_csv", "race_names.csv"))
    df = db.df("SELECT race_name, COUNT(*) AS 件数 FROM races WHERE race_name IS NOT NULL GROUP BY race_name")
    if df.empty:
        return df
    return df[[race_level(n, over) is None for n in df["race_name"]]].sort_values("件数", ascending=False)


def _eval_path(cfg) -> Path:
    return cfg.path("db").parent / "evaluation.json"


def _load_eval(cfg) -> dict | None:
    p = _eval_path(cfg)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def print_tables(db: DB, table: pd.DataFrame | None = None):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    if table is not None and not table.empty:
        print("\n【1】的中率で比べる（検証期間）")
        t = table.copy()
        for c in t.columns:
            if c.endswith("率"):
                t[c] = t[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "-")
            elif t[c].dtype.kind == "f":
                t[c] = t[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "-")
        print(t.to_string(index=False))
    s = summary_table(db.df("SELECT * FROM bets WHERE status IN ('won','lost')"))
    if s.empty:
        print("\n（まだ精算済みの仮想購入はありません）")
        return
    print("\n【2】回収率で比べる（仮想購入）  買い方 / 券種 : 点数  的中率  回収率  [95%区間]  判定")
    for r in s.sort_values(["区分", "券種", "戦略"]).to_dict("records"):
        print(f"  {r['戦略']} / {r['券種']} : {r['点数']}点  {r['的中率']:.1%}  {r['回収率']:.1%}"
              f"  [{r['95%区間 下限']:.0%}〜{r['95%区間 上限']:.0%}]  {r['判定']}")


def _open_file(path: Path):
    if sys.platform == "win32":
        import os
        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except OSError:
            pass


# ---------------- コマンド ----------------
def cmd_demo(cfg, args):
    db_path = ROOT / "data" / "demo.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    db = DB(db_path)
    from .sources.sample import SampleSource
    src = SampleSource(start="2025-01-01")
    d0 = date(2025, 1, 1)
    days = [(d0 + timedelta(days=i)).isoformat() for i in range(args.days)]
    print(f"架空データを {len(days)} 日分作成中…（3日間の大会・勝ち上がり・決まり手つき）")
    for i, d in enumerate(days, 1):
        for race in pipeline.ingest_cards(db, src, d):
            pipeline.ingest_odds(db, src, race)
            pipeline.ingest_result(db, src, race)
        if i % 50 == 0:
            print(f"  {i}/{len(days)} 日")
    n = db.con.execute("SELECT COUNT(*) FROM races").fetchone()[0]
    print(f"  合計 {n} レース")
    out_json = ROOT / "data" / "demo_evaluation.json"
    table = pipeline.evaluate_split(db, cfg, test_frac=args.test_frac, out_json=out_json)
    out = write_report(db, ROOT / "output" / "デモ_検証レポート.xlsx",
                       json.loads(out_json.read_text(encoding="utf-8")), unknown_race_names(db, cfg))
    print_tables(db, table)
    print(f"\nExcel を作成しました: {out}")
    print("※ 架空データでの数字です。どの予想方法が良いかの判断には使えません（動作確認用）。")
    db.close()
    _open_file(out)


def cmd_evaluate(cfg, args):
    db = DB(cfg.path("db"))
    table = pipeline.evaluate_split(db, cfg, test_frac=args.test_frac, allow_final_odds=args.allow_final_odds,
                                    out_json=_eval_path(cfg))
    out = write_report(db, cfg.path("excel"), _load_eval(cfg), unknown_race_names(db, cfg))
    print_tables(db, table)
    print(f"\nExcel: {out}")
    db.close()


def cmd_import(cfg, args):
    db = DB(cfg.path("db"))
    print(import_folder(db, cfg.base / cfg.section("source").get("csv_inbox", "inbox")))
    db.close()


def cmd_fetch(cfg, args):
    db = DB(cfg.path("db"))
    src = make_source(cfg)
    if src is None:
        print("config.toml の [source] name にデータ取得元を設定してください")
        return
    d, end = date.fromisoformat(args.date_from), date.fromisoformat(args.date_to)
    while d <= end:
        ds = d.isoformat()
        try:
            for race in pipeline.ingest_cards(db, src, ds):
                pipeline.ingest_result(db, src, race)
        except Exception as e:
            log.warning("%s の取得に失敗: %s", ds, e)
        d += timedelta(days=1)
    db.close()


def _models_path(cfg) -> Path:
    return cfg.path("db").parent / "models.pkl"


def cmd_daily(cfg, args):
    db = DB(cfg.path("db"))
    if cfg.section("source").get("name") == "csv":
        print(import_folder(db, cfg.base / cfg.section("source").get("csv_inbox", "inbox")))
    src = make_source(cfg)
    today = args.date or date.today().isoformat()
    if src is not None:
        got = 0
        for r in pipeline.pending_result_races(db, today):
            try:
                got += pipeline.ingest_result(db, src, r)
            except Exception as e:
                log.warning("結果取得失敗 %s: %s", r["race_id"], e)
        log.info("結果を %d レース取得", got)
    log.info("精算 %d 点", pipeline.settle(db))
    feats = pipeline.features_for(db, cfg)
    done = feats[feats["finish"].notna()]
    if done["race_id"].nunique() >= cfg.section("model").get("min_train_races", 300):
        methods = pipeline.train_methods(done, pipeline.method_names(cfg))
        _models_path(cfg).write_bytes(pickle.dumps(methods))
    out = write_report(db, cfg.path("excel"), _load_eval(cfg), unknown_race_names(db, cfg))
    log.info("Excel を更新しました: %s", out)
    print_tables(db)
    db.close()


def cmd_predict(cfg, args):
    db = DB(cfg.path("db"))
    src = make_source(cfg)
    today = args.date or date.today().isoformat()
    mp = _models_path(cfg)
    if not mp.exists():
        print("学習済みのAIがありません。先に daily か evaluate を実行してください。")
        return
    methods = pickle.loads(mp.read_bytes())
    if src is not None and not db.con.execute("SELECT 1 FROM races WHERE date=? LIMIT 1", (today,)).fetchone():
        pipeline.ingest_cards(db, src, today)
    races = db.df("SELECT * FROM races WHERE date=? ORDER BY start_time, venue, race_no", (today,)).to_dict("records")
    already = {r[0] for r in db.con.execute("SELECT DISTINCT race_id FROM bets WHERE mode='live' AND date=?", (today,))}
    now = datetime.now()
    window = cfg.section("schedule").get("predict_window_min", 40)
    targets = []
    for r in races:
        if r["race_id"] in already or r["has_result"]:
            continue
        if r.get("start_time"):
            st = datetime.strptime(f"{r['date']} {r['start_time']}", "%Y-%m-%d %H:%M")
            if st <= now or (not args.all and (st - now) > timedelta(minutes=window)):
                continue
        targets.append(r)
    if not targets:
        log.info("予想対象のレースはありません（%s）", today)
        return
    if src is not None:
        for r in targets:
            try:
                pipeline.ingest_odds(db, src, r)
            except Exception as e:
                log.warning("オッズ取得失敗 %s: %s", r["race_id"], e)
    feats = pipeline.features_for(db, cfg, until_date=today)
    n = pipeline.predict_and_bet(db, cfg, methods, feats, [r["race_id"] for r in targets], "live",
                                 decided_at=now_str(), save_predictions_of="LightGBM")
    log.info("%dレースを予想し、仮想購入を %d 点記録しました", len(targets), n)
    db.close()


def cmd_report(cfg, args):
    db = DB(cfg.path("db"))
    print(write_report(db, cfg.path("excel"), _load_eval(cfg), unknown_race_names(db, cfg)))
    db.close()


def main(argv=None):
    p = argparse.ArgumentParser(prog="keirin", description="競輪予想AI（設計書版・仮想購入・回収率記録）")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--days", type=int, default=200)
    d.add_argument("--test-frac", type=float, default=0.22)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--test-frac", type=float, default=0.22, help="新しい方から何割を検証に使うか（初期値 0.22 ≒ 4500中1000）")
    ev.add_argument("--allow-final-odds", action="store_true",
                    help="締切前オッズが無いデータ用。確定オッズで判断する（回収率は甘くなる＝参考値）")
    sub.add_parser("import-csv")
    fe = sub.add_parser("fetch")
    fe.add_argument("--from", dest="date_from", required=True)
    fe.add_argument("--to", dest="date_to", required=True)
    pr = sub.add_parser("predict")
    pr.add_argument("--date")
    pr.add_argument("--all", action="store_true")
    da = sub.add_parser("daily")
    da.add_argument("--date")
    sub.add_parser("report")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg.path("log"))
    {"demo": cmd_demo, "evaluate": cmd_evaluate, "import-csv": cmd_import, "fetch": cmd_fetch,
     "predict": cmd_predict, "daily": cmd_daily, "report": cmd_report}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
