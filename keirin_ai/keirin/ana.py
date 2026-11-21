"""穴党の系統（2026-11-22 に固定。ver1 とは別の系統として記録して比べる）。

穴v1（3連単の中穴）:
  3連単の各目の当たる確率を、3連単の市場確率 q（1/オッズ をレース内で合計1）・オッズ・9車かどうかと、
  展開の特徴（頭が先頭/番手/単騎、1-2着・1-2-3着が同じライン、格上番手の不発、点数差の小さい自力の逆転）から
  ロジスティック回帰で出す（係数は data/market_models/ana_v1.pkl、9/18〜10/25 の確定オッズで学習）。
  20〜100倍の目のうち期待値が一番高い1点を、期待値1.0以上なら買う。
  検証（確定オッズ）: 学習 122%（148点・的中6）、確認 10/26〜11/20 114%（113点・的中4）。的中が少なく未確定。

玉砕番手v1:
  玉砕型の先頭 = ラインの先頭で、過去に先頭で5回以上走り、B（バック先頭）を取った割合が4割以上、
  Bを3回以上取ってそのうち4着以下が5割以上、かつ今回の1着人気が3番目以下。
  その番手（1着人気4番目以下）を頭に、2・3着は頭以外の競走得点上位4人（先頭も入れる）の12点。
  検証（確定オッズ、6/11〜11/20）: 112%（488R・的中17）だが前半54%・後半199%と不安定。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANA_NAME = "穴v1（3連単中穴・期待値1.0）"
GYOKUSAI_NAME = "玉砕番手v1（3連単12点）"


def _norm(d: dict[str, float]) -> dict[str, float]:
    inv = {k: 1.0 / v for k, v in d.items() if v and v > 0}
    s = sum(inv.values())
    return {k: v / s for k, v in inv.items()} if s else {}


def race_shape(entries: pd.DataFrame) -> tuple[dict, dict, set, set]:
    """entries: car_no, score, line_id, line_pos, line_size → (位置, ライン, 格上番手のラインの車, 点差小の逆転の車)。"""
    g = entries.set_index("car_no")
    pos, line = {}, {}
    for car, r in g.iterrows():
        sz = r.line_size if pd.notna(r.line_size) else 1
        pos[int(car)] = "単騎" if sz <= 1 else ("先頭" if r.line_pos == 1 else ("番手" if r.line_pos == 2 else "3番手"))
        line[int(car)] = r.line_id if sz > 1 else -int(car)
    sc = pd.to_numeric(g.score, errors="coerce")
    multi = g[g.line_size.fillna(1) >= 2]
    s1 = set()
    for _, m in multi.groupby("line_id"):
        m = m.sort_values("line_pos")
        if len(m) >= 2 and sc[m.index[1]] - sc[m.index[0]] >= 3:
            s1 |= {int(x) for x in m.index}
    ld = sc[multi[multi.line_pos == 1].index].sort_values(ascending=False)
    s3 = {int(ld.index[1])} if len(ld) >= 2 and ld.iloc[0] - ld.iloc[1] < 2 else set()
    return pos, line, s1, s3


def ana_pick(tri_odds: dict[str, float], entries: pd.DataFrame, model: dict,
             lo: float = 20.0, hi: float = 100.0, min_ev: float = 1.0) -> dict | None:
    """穴v1: 3連単のオッズ（締切前）と出走表 → 買う1点 {combo, odds, pm, ev}。無ければ None（見送り）。"""
    q = _norm(tri_odds)
    if not q:
        return None
    pos, line, s1, s3 = race_shape(entries)
    nine = float(len(entries) == 9)
    rows = []
    for k, p in q.items():
        a, b, c = map(int, k.split(">"))
        f = [pos.get(a) == "先頭", pos.get(a) == "番手", pos.get(a) == "単騎", line.get(a) == line.get(b),
             line.get(a) == line.get(b) == line.get(c), bool(s1) and a not in s1, a in s3]
        rows.append([k, tri_odds[k], np.log(p), np.log(tri_odds[k]), nine] + [float(x) for x in f])
    t = pd.DataFrame(rows, columns=["combo", "odds", "lq", "lo", "nine"] + [f"f{i}" for i in range(7)])
    X = t[["lq", "lo", "nine"] + [f"f{i}" for i in range(7)]].to_numpy(float)
    Z = np.column_stack([np.ones(len(X)), (X - model["mu"]) / model["sd"]])
    t["pm"] = 1 / (1 + np.exp(-(Z @ model["w"])))
    t["ev"] = t.pm * t.odds
    c = t[(t.odds >= lo) & (t.odds <= hi)]
    if c.empty:
        return None
    best = c.loc[c.ev.idxmax()]
    return {"combo": best.combo, "odds": float(best.odds), "pm": float(best.pm), "ev": float(best.ev)} \
        if best.ev >= min_ev else None


def gyokusai_bets(con, race_id: str, date: str, entries: pd.DataFrame, exacta_odds: dict[str, float]) -> list[str]:
    """玉砕番手v1: 条件に合えば3連単の買い目（12点）を返す。合わなければ []。成績は date より前だけ使う。"""
    q = _norm(exacta_odds)
    win = {}
    for k, v in q.items():
        h = int(k.split(">")[0])
        win[h] = win.get(h, 0.0) + v
    if not win:
        return []
    rank = {car: i + 1 for i, (car, _) in enumerate(sorted(win.items(), key=lambda x: -x[1]))}
    g = entries.copy()
    g["score"] = pd.to_numeric(g.score, errors="coerce")
    out = []
    for _, L in g[(g.line_size.fillna(1) >= 2) & (g.line_pos == 1)].iterrows():
        ban = g[(g.line_id == L.line_id) & (g.line_pos == 2)]
        if ban.empty or rank.get(int(L.car_no), 99) < 3:
            continue
        hist = con.execute(
            """SELECT e.finish, i.sb FROM entries e JOIN races r USING(race_id)
               LEFT JOIN result_info i USING(race_id, car_no)
               WHERE e.rider_id=? AND r.date<? AND e.line_pos=1 AND e.line_size>=2 AND e.finish IS NOT NULL""",
            (L.rider_id, date)).fetchall()
        n = len(hist)
        b = [(f, sb) for f, sb in hist if sb and "B" in sb]
        tare = sum(1 for f, _ in b if f is not None and int(f) >= 4)
        if n < 5 or len(b) / n < 0.4 or len(b) < 3 or tare / len(b) < 0.5:
            continue
        h = int(ban.car_no.iloc[0])
        if rank.get(h, 0) < 4:
            continue
        cand = [int(x) for x in g[g.car_no != h].sort_values("score", ascending=False).car_no[:4]]
        out += [f"{h}>{a}>{c}" for a in cand for c in cand if a != c]
    return out
