"""仮想購入のルール（どの組み合わせを何点買うか）。

- AI本命  : 各券種で、AIの確率が一番高い組み合わせを1点（＝「予想通りに買う」）。オッズ不要。
- AI期待値: 確率 × オッズ ≥ min_ev の組み合わせだけ。オッズ必要。
- 人気1位 : 各券種で一番オッズが低い組み合わせを1点。AIと比べるための物差し。
"""
from __future__ import annotations

from .combos import combo_probs

HONMEI, EV, BASELINE = "AI本命", "AI期待値", "人気1位"
STRATEGIES = (HONMEI, EV, BASELINE)


def decide_bets(cars: list[int], win_probs, odds_by_type: dict[str, dict[str, float]],
                odds_is_final: dict[str, bool], cfg: dict, lambda2: float = 1.0, lambda3: float = 1.0) -> list[dict]:
    probs = combo_probs(cars, win_probs, lambda2, lambda3)
    stake = int(cfg.get("stake", 100))
    out: list[dict] = []
    for bt in cfg.get("bet_types", ["2車複", "2車単", "3連複"]):
        pb = probs.get(bt, {})
        if not pb:
            continue
        od = odds_by_type.get(bt, {})
        final = bool(odds_is_final.get(bt, False))

        def bet(strategy, combo):
            o = od.get(combo)
            return {"strategy": strategy, "bet_type": bt, "combo": combo, "stake": stake,
                    "prob": float(pb.get(combo, 0.0)), "odds": o,
                    "ev": float(pb.get(combo, 0.0) * o) if o else None,
                    "odds_final": int(final) if o else None}

        if cfg.get("honmei_enabled", True):
            out.append(bet(HONMEI, max(pb, key=pb.get)))
        if od and cfg.get("ev_enabled", True):
            cands = []
            for c, p in pb.items():
                o = od.get(c)
                if not o or o > cfg.get("max_odds", 150) or p < cfg.get("min_prob", 0.02):
                    continue
                if p * o >= cfg.get("min_ev", 1.2):
                    cands.append((p * o, c))
            for _, c in sorted(cands, reverse=True)[: int(cfg.get("max_bets_per_race", 3))]:
                out.append(bet(EV, c))
        if od and cfg.get("baseline_enabled", True):
            out.append(bet(BASELINE, min(od, key=od.get)))
    return out
