"""コマンド一覧（python -m keirin <コマンド>）

  demo        架空データで「学習→過去検証→Excel」まで一通り試す（ネット不要）
  predict     今日の発走前レースを予想して仮想購入（30分ごとの自動実行用）
  daily       結果取得→精算→AI再学習→Excel更新（1日1回の自動実行用）
  backtest    過去データで検証（--from 2026-01-01 --to 2026-06-30）
  import-csv  inbox フォルダのCSVを取り込む
  train       AIを学習し直す
  report      Excelだけ作り直す
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from . import pipeline
from .config import ROOT, load_config
from .db import DB, now_str
from .model import WinProbModel
from .report import write_report
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


def train_with_metrics(db: DB, cfg, holdout_days: int = 30) -> WinProbModel | None:
    """直近 holdout_days 日を使わずに学習して成績を測り（=本当の実力の目安）、最後に全データで学習し直す。"""
    feats = pipeline.features_for(db, cfg)
    done = feats[feats["finish"].notna()]
    if done.empty:
        return None
    dates = sorted(done["date"].unique())
    metrics = {}
    if len(dates) > holdout_days + 30:
        cut = dates[-holdout_days]
        m = pipeline.train(db, cfg, feats, before_date=cut)
        if m is not None:
            test = done[done["date"] >= cut]
            metrics = {f"直近{holdout_days}日 " + k: v for k, v in m.evaluate(test, m.predict(test)).items()}
    model = pipeline.train(db, cfg, feats)
    if model is not None:
        model.metrics = metrics
    return model


def cmd_demo(cfg, args):
    db_path = ROOT / "data" / "demo.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    db = DB(db_path)
    from .sources.sample import SampleSource
    src = SampleSource()
    end = date.today() - timedelta(days=1)
    days = [(end - timedelta(days=i)).isoformat() for i in range(args.days - 1, -1, -1)]
    print(f"架空データを {len(days)} 日分作成中…")
    for i, d in enumerate(days, 1):
        for race in pipeline.ingest_cards(db, src, d):
            pipeline.ingest_odds(db, src, race)
            pipeline.ingest_result(db, src, race)
        if i % 30 == 0:
            print(f"  {i}/{len(days)} 日")
    test_from = days[-args.test_days]
    print(f"過去検証: {test_from} 〜 {days[-1]}（各日、その日より前のデータだけで学習）")
    res = pipeline.backtest(db, cfg, test_from, days[-1],
                            progress=lambda i, n, d: print(f"  {i}/{n} {d}") if i % 10 == 0 else None)
    print(f"  仮想購入 {res['bets']} 点")
    model = train_with_metrics(db, cfg)
    out = write_report(db, ROOT / "output" / "デモ_回収率レポート.xlsx", model)
    _print_summary(db)
    print(f"\nExcel を作成しました: {out}")
    print("※ 架空データでの数字です。本物の競輪の成績とは関係ありません。")
    db.close()
    _open_file(out)


def _open_file(path: Path):
    """Windows なら作った Excel をそのまま開く。"""
    if sys.platform == "win32":
        import os
        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except OSError:
            pass


def _print_summary(db: DB):
    from .report import summary_table
    s = summary_table(db.df("SELECT * FROM bets WHERE status IN ('won','lost')"))
    if s.empty:
        print("（まだ精算済みの仮想購入はありません）")
        return
    print("\n区分 / 戦略 / 券種 : 点数  的中率  回収率  [95%区間]  判定")
    for r in s.to_dict("records"):
        print(f"{r['区分']} / {r['戦略']} / {r['券種']} : {r['点数']}点  {r['的中率']:.1%}  {r['回収率']:.1%}"
              f"  [{r['95%区間 下限']:.0%}〜{r['95%区間 上限']:.0%}]  {r['判定']}")


def cmd_predict(cfg, args):
    db = DB(cfg.path("db"))
    src = make_source(cfg)
    today = args.date or date.today().isoformat()
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
            if st <= now:
                continue  # 発走後は買えない
            if not args.all and (st - now) > timedelta(minutes=window):
                continue
        targets.append(r)
    if not targets:
        log.info("予想対象のレースはありません（%s）", today)
        return
    if src is not None:
        for r in targets:
            try:
                pipeline.ingest_odds(db, src, r)
            except Exception as e:  # オッズが取れなくても本命予想は続ける
                log.warning("オッズ取得失敗 %s: %s", r["race_id"], e)
    model = WinProbModel.load(cfg.path("model"))
    feats = pipeline.features_for(db, cfg, until_date=today)
    n = pipeline.predict_and_bet(db, cfg, model if model.coef is not None else None, feats,
                                 [r["race_id"] for r in targets], "live", decided_at=now_str())
    log.info("%dレースを予想し、仮想購入を %d 点記録しました", len(targets), n)
    db.close()


def cmd_daily(cfg, args):
    db = DB(cfg.path("db"))
    src = make_source(cfg)
    if cfg.section("source").get("name") == "csv":
        print(import_folder(db, cfg.base / cfg.section("source").get("csv_inbox", "inbox")))
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
    model = train_with_metrics(db, cfg)
    if model is not None:
        model.save(cfg.path("model"))
        log.info("AIを再学習しました（%dレース）", model.trained_races)
    out = write_report(db, cfg.path("excel"), model)
    log.info("Excel を更新しました: %s", out)
    _print_summary(db)
    db.close()


def cmd_backtest(cfg, args):
    db = DB(cfg.path("db"))
    res = pipeline.backtest(db, cfg, args.date_from, args.date_to, allow_final_odds=args.allow_final_odds,
                            progress=lambda i, n, d: print(f"  {i}/{n} {d}") if i % 10 == 0 else None)
    log.info("過去検証: %s", res)
    write_report(db, cfg.path("excel"), WinProbModel.load(cfg.path("model")))
    _print_summary(db)
    db.close()


def cmd_import(cfg, args):
    db = DB(cfg.path("db"))
    print(import_folder(db, cfg.base / cfg.section("source").get("csv_inbox", "inbox")))
    db.close()


def cmd_train(cfg, args):
    db = DB(cfg.path("db"))
    model = train_with_metrics(db, cfg)
    if model is None:
        print("学習データが足りません")
    else:
        model.save(cfg.path("model"))
        print(model.coef_table().to_string(index=False))
        print(model.metrics)
    db.close()


def cmd_report(cfg, args):
    db = DB(cfg.path("db"))
    print(write_report(db, cfg.path("excel"), WinProbModel.load(cfg.path("model"))))
    db.close()


def main(argv=None):
    p = argparse.ArgumentParser(prog="keirin", description="競輪予想AI（仮想購入・回収率記録）")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--days", type=int, default=150)
    d.add_argument("--test-days", type=int, default=60)
    pr = sub.add_parser("predict")
    pr.add_argument("--date")
    pr.add_argument("--all", action="store_true", help="発走時刻に関係なく、未予想のレースを全部予想")
    da = sub.add_parser("daily")
    da.add_argument("--date")
    bt = sub.add_parser("backtest")
    bt.add_argument("--from", dest="date_from", required=True)
    bt.add_argument("--to", dest="date_to", required=True)
    bt.add_argument("--allow-final-odds", action="store_true",
                    help="締切前オッズが無いデータ用。確定オッズで判断する（結果は甘くなる＝参考値）")
    sub.add_parser("import-csv")
    sub.add_parser("train")
    sub.add_parser("report")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg.path("log"))
    {"demo": cmd_demo, "predict": cmd_predict, "daily": cmd_daily, "backtest": cmd_backtest,
     "import-csv": cmd_import, "train": cmd_train, "report": cmd_report}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
