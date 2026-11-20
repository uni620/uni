"""Excel レポート（回収率の毎日記録）。"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.chart import LineChart, Reference
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .db import DB
from .strategy import min_odds

log = logging.getLogger(__name__)

HEADER_FILL = PatternFill("solid", fgColor="1F3864")
GOOD = PatternFill("solid", fgColor="C6EFCE")
BAD = PatternFill("solid", fgColor="FFC7CE")
MODE_JA = {"live": "本番（仮想購入）", "backtest": "過去検証"}


def bootstrap_ci(returns: np.ndarray, stakes: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """回収率の95%区間。「この範囲なら運でも普通に起きる」という幅。"""
    if len(returns) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(returns), size=(n, len(returns)))
    r = returns[idx].sum(axis=1) / stakes[idx].sum(axis=1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def judge(lo: float, hi: float, n: int) -> str:
    if n < 30 or np.isnan(lo):
        return "データ不足"
    if lo > 1.0:
        return "◎ 100%超え（運だけでは説明しにくい）"
    if hi < 1.0:
        return "× 100%未満（負けている）"
    return "△ まだ判断できない（運の範囲）"


def summary_table(bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if bets.empty:
        return pd.DataFrame()
    for (mode, strat, bt), g in bets.groupby(["mode", "strategy", "bet_type"]):
        stake, pay = g["stake"].to_numpy(float), g["payout"].to_numpy(float)
        lo, hi = bootstrap_ci(pay, stake)
        rows.append({
            "区分": MODE_JA.get(mode, mode), "戦略": strat, "券種": bt, "点数": len(g),
            "的中": int((g["status"] == "won").sum()), "的中率": (g["status"] == "won").mean(),
            "投資(円)": int(stake.sum()), "払戻(円)": int(pay.sum()), "収支(円)": int(pay.sum() - stake.sum()),
            "回収率": pay.sum() / stake.sum(), "95%区間 下限": lo, "95%区間 上限": hi,
            "判定": judge(lo, hi, len(g)),
            "確定オッズで判断(参考値)": int((g["odds_final"] == 1).sum()),
        })
    return pd.DataFrame(rows)


def daily_table(bets: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()
    d = bets.groupby(["mode", "strategy", "bet_type", "date"]).agg(
        点数=("stake", "size"), 的中=("status", lambda s: int((s == "won").sum())),
        投資=("stake", "sum"), 払戻=("payout", "sum")).reset_index().sort_values("date")
    d["回収率"] = d["払戻"] / d["投資"]
    g = d.groupby(["mode", "strategy", "bet_type"])
    d["累計投資"] = g["投資"].cumsum()
    d["累計払戻"] = g["払戻"].cumsum()
    d["累計回収率"] = d["累計払戻"] / d["累計投資"]
    d["mode"] = d["mode"].map(MODE_JA).fillna(d["mode"])
    return d.rename(columns={"mode": "区分", "strategy": "戦略", "bet_type": "券種", "date": "日付"}) \
            .sort_values(["日付", "区分", "戦略", "券種"], ascending=[False, True, True, True])


def cumulative_pivot(bets: pd.DataFrame, mode: str) -> pd.DataFrame:
    b = bets[bets["mode"] == mode]
    if b.empty:
        return pd.DataFrame()
    b = b.assign(key=b["strategy"])  # 券種はまとめて、買い方ごとに1本の線
    d = b.groupby(["date", "key"])[["stake", "payout"]].sum().unstack("key").fillna(0).sort_index()
    cum = d["payout"].cumsum() / d["stake"].cumsum().replace(0, np.nan)
    return cum.reset_index().rename(columns={"date": "日付"})


def _style(ws, df: pd.DataFrame, pct_cols=(), yen_cols=(), start_row: int = 1):
    for c in range(1, ws.max_column + 1):
        cell = ws.cell(row=start_row, column=c)
        cell.fill, cell.font = HEADER_FILL, Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cols = list(df.columns)
    for name in pct_cols:
        if name in cols:
            j = cols.index(name) + 1
            for r in range(start_row + 1, ws.max_row + 1):
                ws.cell(row=r, column=j).number_format = "0.0%"
    for name in yen_cols:
        if name in cols:
            j = cols.index(name) + 1
            for r in range(start_row + 1, ws.max_row + 1):
                ws.cell(row=r, column=j).number_format = "#,##0"
    for j, name in enumerate(cols, start=1):
        width = max([len(str(name))] + [len(str(v)) for v in df[name].head(200)]) if len(df) else len(str(name))
        ws.column_dimensions[get_column_letter(j)].width = min(48, max(9, width * 1.6))
    ws.freeze_panes = ws.cell(row=start_row + 1, column=1)


def _color_recovery(ws, df: pd.DataFrame, col: str, start_row: int = 1):
    if col not in df.columns or ws.max_row <= start_row:
        return
    L = get_column_letter(list(df.columns).index(col) + 1)
    rng = f"{L}{start_row + 1}:{L}{ws.max_row}"
    ws.conditional_formatting.add(rng, CellIsRule(operator="greaterThanOrEqual", formula=["1"], fill=GOOD))
    ws.conditional_formatting.add(rng, CellIsRule(operator="lessThan", formula=["1"], fill=BAD))


EXPLAIN = [
    ("このファイルの見方", ""),
    ("回収率", "払戻 ÷ 投資。100%を超えればプラス。競輪は払戻率が約75%なので、適当に買い続けると75%前後に近づく。"),
    ("95%区間", "運のブレを考えた回収率の幅。下限が100%を超えて初めて「勝てている可能性が高い」と言える。"),
    ("判定", "◎=運だけでは説明しにくいプラス / △=まだ運の範囲 / ×=負けている / データ不足=30点未満"),
    ("予想方法", "人気順 / 手作り7:3（強さ7・調子3、AIなし）/ LightGBM / LightGBM+競走得点 の4つを比べる（設計書どおり）。"),
    ("〇〇・本命", "その予想方法で確率が一番高い組み合わせを各券種1点。『予想通りに買ったときの回収率』はこれ。"),
    ("〇〇・期待値", "確率×オッズが基準（初期値1.2）以上の組み合わせだけ買う。オッズが取れたレースだけ。"),
    ("人気1位", "一番人気（最低オッズ）を買い続けた場合。これに勝てないなら、予想方法を使う意味は薄い。"),
    ("1位予想の1着率", "その方法で1着確率が一番高い選手が、実際に1着だった割合。回収率より安定していて比べやすい。"),
    ("予想のズレ", "1着になった選手に、どれだけ低い確率をつけていたか。小さいほど良い。でたらめ予想より小さくないと意味がない。"),
    ("AIが重視した列", "LightGBM がどの列を判断に多く使ったか（合計100%）。あなたの感覚と違うところは要確認。"),
    ("格が不明なレース名", "格を自動で決められなかったレース名。race_names.csv に点数を書くと次回から使われる。"),
    ("本番 / 過去検証", "本番=毎日の自動仮想購入。過去検証=過去データでの試算（本番より甘く出やすいので参考程度）。"),
    ("確定オッズで判断(参考値)", "締切後のオッズを使って判断した点数。実際には買えないタイミングなので、多いほど結果は甘い。"),
    ("〇〇・3連複ずらし", "3連複で確率1番の組み合わせは人気が集まり配当が安いので、2〜4番目の3点を買う。過去検証で1番だけより回収率が高かった。"),
    ("〇〇・絞り込み", "本命のうち、成績の悪い3連複とA級チャレンジのレースを外したもの（2車単・2車複だけ）。"),
    ("買う目安オッズ", "AIの確率×オッズが1.2以上になる最低の倍率。アプリで見たオッズがこれ未満なら、期待値が足りないので見送りが目安。損得なしオッズは1.0倍のライン。2車単・2車複はAIの確率がほぼ正確。3連複はAIの確率が高いほど出しすぎるので、実際の的中率に直して計算している。保証ではない。"),
    ("注意", "これは記録用の仮想購入で、実際の購入はしていない。架空データ（sample）での数字は本物の競輪と無関係。"),
]


def write_report(db: DB, path: str | Path, eval_info: dict | None = None,
                 unknown_names: pd.DataFrame | None = None) -> Path:
    path = Path(path)
    bets = db.df("SELECT * FROM bets WHERE status IN ('won','lost')")
    all_bets = db.df("SELECT * FROM bets ORDER BY decided_at DESC, id DESC LIMIT 5000")
    summ = summary_table(bets)
    daily = daily_table(bets)
    cum_mode = "live" if (bets["mode"] == "live").any() else "backtest"
    cum = cumulative_pivot(bets, cum_mode)

    last_date = db.con.execute("SELECT MAX(r.date) FROM predictions p JOIN races r USING (race_id)").fetchone()[0]
    preds = db.df("""SELECT r.date AS 日付, r.venue AS 場, r.race_no AS R, r.race_name AS レース名,
                            r.start_time AS 発走, e.car_no AS 車番, e.name AS 選手,
                            e.line_id AS ライン, e.line_pos AS 並び, p.win_prob AS AI勝率, e.finish AS 着順
                     FROM predictions p JOIN races r USING (race_id)
                     JOIN entries e ON e.race_id=p.race_id AND e.car_no=p.car_no
                     WHERE r.date=? ORDER BY r.venue, r.race_no, p.win_prob DESC""", (last_date,)) \
        if last_date else pd.DataFrame()
    if not preds.empty:
        preds.insert(preds.columns.get_loc("AI勝率") + 1, "AI順位",
                     preds.groupby(["場", "R"])["AI勝率"].rank(ascending=False, method="first").astype(int))

    # 今日の買い目と「この倍率未満なら買わない」目安（AIの確率×オッズ ≥ 1.2 になる最低オッズ）
    margin = 1.2
    today = db.df("""SELECT r.venue AS 場, r.race_no AS R, r.start_time AS 発走, b.strategy AS 戦略,
                            b.bet_type AS 券種, b.combo AS 買い目, b.prob AS AI確率, b.status AS 結果, b.payout AS 払戻
                     FROM bets b JOIN races r USING (race_id)
                     WHERE b.mode='live' AND b.date=(SELECT MAX(date) FROM bets WHERE mode='live')
                     ORDER BY r.start_time, r.venue, r.race_no, b.strategy, b.bet_type""")
    if not today.empty:
        today["損得なしオッズ"] = [min_odds(p, 1.0, bt) for p, bt in zip(today["AI確率"], today["券種"])]
        today["買う目安オッズ(これ未満は見送り)"] = [min_odds(p, margin, bt) for p, bt in zip(today["AI確率"], today["券種"])]
        today["結果"] = today["結果"].map({"won": "的中", "lost": "外れ", "refund": "返還", "open": "未確定"})
    hist = all_bets.rename(columns={
        "mode": "区分", "race_id": "レースID", "date": "日付", "strategy": "戦略", "bet_type": "券種",
        "combo": "買い目", "stake": "金額", "prob": "AI確率", "odds": "オッズ", "odds_final": "確定オッズ使用",
        "ev": "期待値", "decided_at": "判断時刻", "status": "結果", "payout": "払戻", "settled_at": "精算時刻"})
    if not hist.empty:
        hist["区分"] = hist["区分"].map(MODE_JA).fillna(hist["区分"])
        hist["結果"] = hist["結果"].map({"won": "的中", "lost": "外れ", "refund": "返還", "open": "未確定"})
        hist = hist.drop(columns=["id"])

    info = [("更新日時", datetime.now().strftime("%Y-%m-%d %H:%M"))]
    comp = pd.DataFrame()
    imp = pd.DataFrame()
    if eval_info:
        info += [("学習に使ったレース", f"{eval_info['train_races']}レース（{eval_info['cut_date']} より前）"),
                 ("検証に使ったレース", f"{eval_info['test_races']}レース（{eval_info['cut_date']} 以降）")]
        comp = pd.DataFrame(eval_info["table"])
        parts = []
        for name, rows in eval_info.get("importance", {}).items():
            t = pd.DataFrame(rows)
            if not t.empty:
                parts.append(t.assign(予想方法=name)[["予想方法", "特徴量", "重要度"]])
        imp = pd.concat(parts) if parts else pd.DataFrame()

    def _write(target: Path):
        with pd.ExcelWriter(target, engine="openpyxl") as xw:
            head = pd.DataFrame(info, columns=["項目", "値"])
            head.to_excel(xw, sheet_name="サマリー", index=False)
            comp_row = len(head) + 3
            if not comp.empty:
                pd.DataFrame({"【1】的中率で比べる（検証期間）": []}).to_excel(
                    xw, sheet_name="サマリー", index=False, startrow=comp_row - 2)
                comp.to_excel(xw, sheet_name="サマリー", index=False, startrow=comp_row - 1)
            start = comp_row + (len(comp) + 4 if not comp.empty else 0)
            pd.DataFrame({"【2】回収率で比べる（仮想購入）": []}).to_excel(
                xw, sheet_name="サマリー", index=False, startrow=start - 2)
            if summ.empty:
                pd.DataFrame({"お知らせ": ["まだ結果の出た仮想購入がありません"]}).to_excel(
                    xw, sheet_name="サマリー", index=False, startrow=start - 1)
            else:
                summ.to_excel(xw, sheet_name="サマリー", index=False, startrow=start - 1)
            (daily if not daily.empty else pd.DataFrame({"日付": []})).to_excel(xw, sheet_name="日別", index=False)
            (cum if not cum.empty else pd.DataFrame({"日付": []})).to_excel(xw, sheet_name="累計回収率グラフ", index=False)
            (today if not today.empty else pd.DataFrame({"場": []})).to_excel(xw, sheet_name="今日の買い目と目安オッズ", index=False)
            (preds if not preds.empty else pd.DataFrame({"日付": []})).to_excel(xw, sheet_name="最新の予想", index=False)
            (hist if not hist.empty else pd.DataFrame({"日付": []})).to_excel(xw, sheet_name="購入履歴", index=False)
            (imp if not imp.empty else pd.DataFrame(columns=["予想方法", "特徴量", "重要度"])).to_excel(
                xw, sheet_name="AIが重視した列", index=False)
            un = unknown_names if unknown_names is not None and not unknown_names.empty else \
                pd.DataFrame({"race_name": [], "件数": []})
            un.to_excel(xw, sheet_name="格が不明なレース名", index=False)
            pd.DataFrame(EXPLAIN, columns=["用語", "説明"]).to_excel(xw, sheet_name="説明", index=False)

            wb = xw.book
            ws = wb["サマリー"]
            ws.column_dimensions["A"].width = 22
            ws.column_dimensions["B"].width = 22
            if not comp.empty:
                _style(ws, comp, pct_cols=[c for c in comp.columns if c.endswith("率")], start_row=comp_row)
            if not summ.empty:
                _style(ws, summ, pct_cols=["的中率", "回収率", "95%区間 下限", "95%区間 上限"],
                       yen_cols=["投資(円)", "払戻(円)", "収支(円)"], start_row=start)
                _color_recovery(ws, summ, "回収率", start_row=start)
                ws.freeze_panes = None
            for name, df, pct, yen in [
                ("日別", daily, ["回収率", "累計回収率"], ["投資", "払戻", "累計投資", "累計払戻"]),
                ("最新の予想", preds, ["AI勝率"], []),
                ("購入履歴", hist, ["AI確率"], ["金額", "払戻"]),
                ("AIが重視した列", imp, ["重要度"], []),
            ]:
                if not df.empty:
                    _style(wb[name], df, pct, yen)
                    if name == "日別":
                        _color_recovery(wb[name], df, "累計回収率")
            ws = wb["累計回収率グラフ"]
            if not cum.empty:
                _style(ws, cum, pct_cols=list(cum.columns[1:]))
                ch = LineChart()
                ch.title = f"累計回収率（{MODE_JA[cum_mode]}）"
                ch.y_axis.title, ch.x_axis.title = "回収率", "日付"
                ch.y_axis.number_format = "0%"
                ch.height, ch.width = 10, 24
                data = Reference(ws, min_col=2, max_col=ws.max_column, min_row=1, max_row=ws.max_row)
                ch.add_data(data, titles_from_data=True)
                ch.set_categories(Reference(ws, min_col=1, min_row=2, max_row=ws.max_row))
                ws.add_chart(ch, f"{get_column_letter(ws.max_column + 2)}2")
            wb["説明"].column_dimensions["A"].width = 26
            wb["説明"].column_dimensions["B"].width = 110

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write(path)
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        log.warning("%s が開かれていて上書きできないため、%s に保存しました", path.name, alt.name)
        _write(alt)
        return alt
