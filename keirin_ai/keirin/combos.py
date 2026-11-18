"""各選手の1着確率 → 2車複・2車単・3連複の的中確率への変換。

ハービル式（Harville）:
    P(i が1着, j が2着) = p_i × p_j / (1 - p_i)
「1着の人を除いた残りで、また同じ強さの比で1着を争う」と考える方法。
lambda2 / lambda3 を 1 未満にすると、2・3着の確率を均す補正（Lo & Bacon-Shone 型）になる。
"""
from __future__ import annotations

import numpy as np

BET_TYPES = ("2車複", "2車単", "3連複")


def combo_key(bet_type: str, cars: tuple[int, ...]) -> str:
    if bet_type == "2車単":
        return f"{cars[0]}>{cars[1]}"
    return "-".join(str(c) for c in sorted(cars))


def combo_cars(combo: str) -> list[int]:
    return [int(x) for x in combo.replace(">", "-").split("-")]


def _cond(p: np.ndarray, lam: float, excluded: list[int]) -> np.ndarray:
    q = np.power(np.clip(p, 1e-12, None), lam)
    q[excluded] = 0.0
    s = q.sum()
    return q / s if s > 0 else q


def combo_probs(cars: list[int], win_probs: list[float] | np.ndarray,
                lambda2: float = 1.0, lambda3: float = 1.0) -> dict[str, dict[str, float]]:
    """cars: 車番リスト, win_probs: 同じ順の1着確率（合計1）。
    戻り値: {券種: {組み合わせ: 確率}}"""
    p = np.asarray(win_probs, dtype=float)
    p = p / p.sum()
    n = len(cars)
    out = {bt: {} for bt in BET_TYPES}
    for i in range(n):
        p2 = _cond(p, lambda2, [i])
        for j in range(n):
            if j == i:
                continue
            pij = p[i] * p2[j]
            k2 = combo_key("2車単", (cars[i], cars[j]))
            out["2車単"][k2] = pij
            k = combo_key("2車複", (cars[i], cars[j]))
            out["2車複"][k] = out["2車複"].get(k, 0.0) + pij
            if n >= 3:
                p3 = _cond(p, lambda3, [i, j])
                for m in range(n):
                    if m in (i, j):
                        continue
                    k3 = combo_key("3連複", (cars[i], cars[j], cars[m]))
                    out["3連複"][k3] = out["3連複"].get(k3, 0.0) + pij * p3[m]
    return out


def implied_win_probs_from_exacta(cars: list[int], exacta_odds: dict[str, float]) -> np.ndarray | None:
    """2車単オッズから、市場が見ている1着確率をざっくり逆算（比較・分析用）。"""
    if not exacta_odds:
        return None
    inv = {k: 1.0 / v for k, v in exacta_odds.items() if v and v > 0}
    tot = sum(inv.values())
    if tot <= 0:
        return None
    w = np.array([sum(v for k, v in inv.items() if k.split(">")[0] == str(c)) for c in cars])
    return w / w.sum() if w.sum() > 0 else None


__all__ = ["BET_TYPES", "combo_key", "combo_cars", "combo_probs", "implied_win_probs_from_exacta"]
