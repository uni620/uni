"""仮想購入のルール（どの組み合わせを何点買うか）。

予想方法（手作り7:3 / LightGBM など）ごとに、次の2つの買い方を記録する。
- 〇〇・本命  : 各券種で、確率が一番高い組み合わせを1点（＝「予想通りに買う」）。オッズ不要。
- 〇〇・期待値: 確率 × オッズ ≥ min_ev の組み合わせだけ。オッズ必要。
比べる物差しとして、どの予想方法にも関係なく次も記録する。
- 人気1位     : 各券種で一番オッズが低い組み合わせを1点。
"""
from __future__ import annotations

from .combos import combo_probs

BASELINE = "人気1位"


def _bet(strategy, bt, combo, stake, prob, od, final):
    o = od.get(combo)
    return {"strategy": strategy, "bet_type": bt, "combo": combo, "stake": stake, "prob": prob,
            "odds": o, "ev": prob * o if (o and prob is not None) else None,
            "odds_final": int(final) if o else None}


def decide_bets(method: str, cars: list[int], win_probs, odds_by_type: dict[str, dict[str, float]],
                odds_is_final: dict[str, bool], cfg: dict, lambda2: float = 1.0, lambda3: float = 1.0,
                probs: dict | None = None) -> list[dict]:
    """probs を渡すと、その組み合わせ確率を使う（ライン連動など）。無ければハービル式で作る。"""
    probs = probs or combo_probs(cars, win_probs, lambda2, lambda3)
    stake = int(cfg.get("stake", 100))
    out: list[dict] = []
    for bt in cfg.get("bet_types", ["2車複", "2車単", "3連複"]):
        pb = probs.get(bt, {})
        if not pb:
            continue
        od = odds_by_type.get(bt, {})
        final = bool(odds_is_final.get(bt, False))
        if cfg.get("honmei_enabled", True):
            c = max(pb, key=pb.get)
            out.append(_bet(f"{method}・本命", bt, c, stake, float(pb[c]), od, final))
        if bt == "3連複" and cfg.get("trio_shift_enabled", True):
            # 3連複は確率1番の組み合わせが極端に弱い（過去検証で回収率60%前後）。2〜4番目の3点にずらす
            ranked = sorted(pb, key=pb.get, reverse=True)[1:4]
            for c in ranked:
                out.append(_bet(f"{method}・3連複ずらし", bt, c, stake, float(pb[c]), od, final))
        if od and cfg.get("ev_enabled", True):
            cands = []
            for c, p in pb.items():
                o = od.get(c)
                if not o or o > cfg.get("max_odds", 150) or p < cfg.get("min_prob", 0.02):
                    continue
                if p * o >= cfg.get("min_ev", 1.2):
                    cands.append((p * o, c))
            for _, c in sorted(cands, reverse=True)[: int(cfg.get("max_bets_per_race", 3))]:
                out.append(_bet(f"{method}・期待値", bt, c, stake, float(pb[c]), od, final))
    return out


def filtered_bets(bets: list[dict], race_class: str | None, cfg: dict) -> list[dict]:
    """「〇〇・本命」から、成績の悪い券種・クラスを外したものを「〇〇・絞り込み」として複製する。"""
    f = cfg.get("filtered", {})
    if not f.get("enabled", True):
        return []
    if race_class in f.get("skip_race_class", ["A級チャレンジ"]):
        return []
    keep = set(f.get("bet_types", ["2車単", "2車複"]))
    return [{**b, "strategy": b["strategy"].replace("・本命", "・絞り込み")}
            for b in bets if b["strategy"].endswith("・本命") and b["bet_type"] in keep]


def skip_overconfident(bets: list[dict], cars: list[int], win_probs, market: dict | None,
                       race_class: str | None, cfg: dict) -> list[dict]:
    """「〇〇・強気見送り」: 本命と同じ買い目。ただし次のレースは買わない。
    - AIの1番手が1番人気と同じで、AIの1着確率 > 市場の支持率（AIが人気以上に強気）
    - A級チャレンジ
    - 人気の情報が無いレース（比べられないので記録しない）
    9/23〜11/17 の検証で、AIが強気のレースは3つの予想方法×3券種すべてで回収率が悪かった（59〜81%）。
    人気は買うかどうかの判断だけに使い、予想そのものには入れない。"""
    f = cfg.get("skip_overconfident", {})
    if not f.get("enabled", True) or not market or market.get("sup1") is None:
        return []
    if race_class in f.get("skip_race_class", ["A級チャレンジ"]):
        return []
    import numpy as np
    i = int(np.argmax(win_probs))
    if int(cars[i]) == int(market["fav1"]) and float(win_probs[i]) > float(market["sup1"]) + f.get("margin", 0.0):
        return []
    return [{**b, "strategy": b["strategy"].replace("・本命", "・強気見送り")}
            for b in bets if b["strategy"].endswith("・本命")]


# 3連複の確率は高いほど出しすぎになる（ハービル式の癖）。9/23〜11/19 の約1.6万点で測った実際の的中率に直す。
# 2車単・2車複はほぼ合っていた（AI 25%→実際24〜27%）ので直さない。
TRIO_CALIB = ([0, .043, .068, .099, .153, .241, .387, 1.0], [0, .043, .067, .086, .115, .140, .178, .25])


def calibrated(prob: float, bet_type: str | None = None) -> float:
    if bet_type == "3連複":
        import numpy as np
        return float(np.interp(prob, *TRIO_CALIB))
    return prob


def min_odds(prob: float | None, margin: float = 1.2, bet_type: str | None = None) -> float | None:
    """この倍率より低ければ買わない方がいい目安（実際の的中率×オッズ ≥ margin になる最低オッズ）。"""
    if not prob or prob <= 0:
        return None
    return round(margin / calibrated(prob, bet_type), 1)


def decide_baseline(odds_by_type: dict[str, dict[str, float]], odds_is_final: dict[str, bool], cfg: dict) -> list[dict]:
    if not cfg.get("baseline_enabled", True):
        return []
    stake = int(cfg.get("stake", 100))
    out = []
    for bt in cfg.get("bet_types", ["2車複", "2車単", "3連複"]):
        od = odds_by_type.get(bt, {})
        if od:
            c = min(od, key=od.get)
            out.append(_bet(BASELINE, bt, c, stake, None, od, bool(odds_is_final.get(bt, False))))
    return out
