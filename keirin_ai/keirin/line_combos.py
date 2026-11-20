"""ライン連動: 1着の確率 → 2着・3着を「誰が1着（2着）に来たか」で変える組み合わせ確率。

今までの combos.py（ハービル式）は、2着・3着を「1着になる力」の比だけで決めていた。
競輪では先頭が勝てば番手が2着に来やすい（10,106レースで 51%。別ラインが勝ったときの番手は 10%）。
そこで 2着・3着の選び方に、ラインのつながりと位置を足す:

  2着の点数 (1着=i のとき j) = a1·log p_j + a2·[i の真後ろ] + a3·[i の真ん前] + a4·[同じラインで離れている]
                              + a5·[番手] + a6·[3番手以降] + a7·[単騎]
  3着の点数 (1・2着=i,j のとき k) = b1·log p_k + b2·[i か j の真後ろ] + b3·[i か j の真ん前] + b4·[同じラインで離れている]
                              + b5·[番手] + b6·[3番手以降] + b7·[単騎]
  それぞれ残りの選手でソフトマックス（合計1）。

係数は過去のレースで当てはめた値（scripts/fit_line_combos.py）。p_j は AI の1着確率（人気・オッズは使わない）。
"""
from __future__ import annotations

import numpy as np

from .combos import BET_TYPES, combo_key

# 9/09〜11/19 の walk-forward 予想（LightGBM+競走得点、4,829レース）で当てはめた値（fit_line_combos.py の出力）。
# 10/14 までで当てはめて 10/15〜11/19 で試したとき、本命の的中率は 3連複 13.2%→23.5%、2車複 26.4%→35.3%。
# 並びは [1着の力, 真後ろ, 真ん前, 同ライン離れ, 番手, 3番手以降, 単騎]
DEFAULT_PARAMS = {
    "a": [0.7308, 1.5654, 1.4437, 0.4713, 0.1143, 0.2679, 0.4389],
    "b": [0.6021, 1.0722, 0.7117, 0.6911, 0.1295, 0.4851, 0.6755],
}
# 係数をすべて「力だけ」にするとハービル式（combos.combo_probs）と同じになる。
HARVILLE_PARAMS = {"a": [1.0, 0, 0, 0, 0, 0, 0], "b": [1.0, 0, 0, 0, 0, 0, 0]}


def relation_features(cars: list[int], lines: dict[int, tuple]) -> tuple[np.ndarray, np.ndarray]:
    """→ (rel[i, j, 3]: j は i の 真後ろ/真ん前/同ライン離れ, own[j, 3]: 番手/3番手以降/単騎)。
    lines: 車番 → (line_id, line_pos, line_size)。不明な車は単騎扱い。"""
    n = len(cars)
    info = [lines.get(c, (None, 1, 1)) for c in cars]
    rel = np.zeros((n, n, 3))
    own = np.zeros((n, 3))
    for j, (lj, pj, sj) in enumerate(info):
        sj = sj or 1
        pj = pj or 1
        if sj <= 1:
            own[j, 2] = 1
        elif pj == 2:
            own[j, 0] = 1
        elif pj >= 3:
            own[j, 1] = 1
    for i, (li, pi, si) in enumerate(info):
        for j, (lj, pj, sj) in enumerate(info):
            if i == j or li is None or lj is None or li != lj or (si or 1) <= 1:
                continue
            if pj == pi + 1:
                rel[i, j, 0] = 1
            elif pj == pi - 1:
                rel[i, j, 1] = 1
            else:
                rel[i, j, 2] = 1
    return rel, own


def ordered_probs(win_probs, rel: np.ndarray, own: np.ndarray, params: dict) -> np.ndarray:
    """→ P[i, j, k]: i が1着・j が2着・k が3着の確率（n×n×n、合計1）。"""
    p = np.clip(np.asarray(win_probs, dtype=float), 1e-9, None)
    p = p / p.sum()
    n = len(p)
    a, b = np.asarray(params["a"], float), np.asarray(params["b"], float)
    lp = np.log(p)
    base2 = a[0] * lp + own @ a[4:7]                     # j ごと
    base3 = b[0] * lp + own @ b[4:7]
    out = np.zeros((n, n, n))
    for i in range(n):
        s2 = base2 + rel[i] @ a[1:4]
        s2[i] = -np.inf
        e2 = np.exp(s2 - s2[np.isfinite(s2)].max())
        q2 = e2 / e2.sum()
        for j in range(n):
            if j == i:
                continue
            r = np.maximum(rel[i], rel[j])               # i か j とのつながり
            s3 = base3 + r @ b[1:4]
            s3[[i, j]] = -np.inf
            if not np.isfinite(s3).any():
                continue
            e3 = np.exp(s3 - s3[np.isfinite(s3)].max())
            out[i, j] = p[i] * q2[j] * e3 / e3.sum()
    return out


def line_combo_probs(cars: list[int], win_probs, lines: dict[int, tuple],
                     params: dict | None = None) -> dict[str, dict[str, float]]:
    """combos.combo_probs と同じ形（{券種: {組み合わせ: 確率}}）で返す。"""
    params = params or DEFAULT_PARAMS
    rel, own = relation_features(cars, lines)
    P = ordered_probs(win_probs, rel, own, params)
    n = len(cars)
    out = {bt: {} for bt in BET_TYPES}
    P2 = P.sum(axis=2) if n >= 3 else None
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pij = float(P2[i, j]) if P2 is not None else 0.0
            out["2車単"][combo_key("2車単", (cars[i], cars[j]))] = pij
            k2 = combo_key("2車複", (cars[i], cars[j]))
            out["2車複"][k2] = out["2車複"].get(k2, 0.0) + pij
            for k in range(n):
                if k in (i, j):
                    continue
                k3 = combo_key("3連複", (cars[i], cars[j], cars[k]))
                out["3連複"][k3] = out["3連複"].get(k3, 0.0) + float(P[i, j, k])
    for k3, v in out["3連複"].items():
        x, y, z = k3.split("-")
        for k in (f"{x}-{y}", f"{x}-{z}", f"{y}-{z}"):
            out["ワイド"][k] = out["ワイド"].get(k, 0.0) + v
    return out


__all__ = ["DEFAULT_PARAMS", "HARVILLE_PARAMS", "relation_features", "ordered_probs", "line_combo_probs"]
