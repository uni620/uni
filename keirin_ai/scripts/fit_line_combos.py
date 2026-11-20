"""ライン連動（keirin/line_combos.py）の係数を、過去レースの実際の 1・2・3着で当てはめる。

使い方: python scripts/fit_line_combos.py 予想.pkl 学習の終わり(YYYY-MM-DD) [予想方法]
予想.pkl: race_id, date, car_no, p（AIの1着確率）, method の表（walk-forward の予想）。
1着の確率は AI のまま。2着・3着の選び方の係数だけを最尤法で決める。
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keirin.line_combos import relation_features  # noqa: E402


def load(pred_path: str, method: str) -> list[dict]:
    t = pd.read_pickle(pred_path)
    t = t[t["method"] == method]
    con = sqlite3.connect(ROOT / "data" / "keirin.sqlite")
    e = pd.read_sql("SELECT race_id, car_no, finish, line_id, line_pos, line_size, scratched FROM entries", con)
    e = e[e["race_id"].isin(set(t["race_id"]))]
    E = {rid: g for rid, g in e.groupby("race_id")}
    races = []
    for rid, g in t.groupby("race_id"):
        g = g.sort_values("car_no")
        ee = E.get(rid)
        if ee is None or (ee["scratched"] == 1).any():
            continue
        fin = dict(zip(ee["car_no"].astype(int), pd.to_numeric(ee["finish"], errors="coerce")))
        cars = g["car_no"].astype(int).tolist()
        f = [fin.get(c) for c in cars]
        try:
            i, j, k = (f.index(x) for x in (1, 2, 3))
        except ValueError:
            continue                                      # 同着・失格などで 1〜3着がそろわない
        lines = {int(r.car_no): (r.line_id, r.line_pos, r.line_size) for r in ee.itertuples()}
        rel, own = relation_features(cars, lines)
        p = np.clip(g["p"].to_numpy(float), 1e-9, None)
        races.append({"race_id": rid, "date": g["date"].iloc[0], "lp": np.log(p / p.sum()),
                      "rel": rel, "own": own, "ijk": (i, j, k), "cars": cars, "p": p / p.sum(), "lines": lines})
    return races


def nll(w: np.ndarray, races: list[dict], stage: int) -> float:
    tot = 0.0
    for r in races:
        i, j, k = r["ijk"]
        if stage == 2:
            s = w[0] * r["lp"] + r["rel"][i] @ w[1:4] + r["own"] @ w[4:7]
            s[i] = -np.inf
            tgt = j
        else:
            s = w[0] * r["lp"] + np.maximum(r["rel"][i], r["rel"][j]) @ w[1:4] + r["own"] @ w[4:7]
            s[[i, j]] = -np.inf
            tgt = k
        m = s[np.isfinite(s)].max()
        tot -= s[tgt] - (m + np.log(np.exp(s[np.isfinite(s)] - m).sum()))
    return tot / len(races)


def fit(races: list[dict]) -> dict:
    out = {}
    for stage, key in ((2, "a"), (3, "b")):
        w0 = np.array([1.0, 0, 0, 0, 0, 0, 0])
        res = minimize(nll, w0, args=(races, stage), method="L-BFGS-B")
        out[key] = [round(float(x), 4) for x in res.x]
    return out


if __name__ == "__main__":
    pred, cut = sys.argv[1], sys.argv[2]
    method = sys.argv[3] if len(sys.argv) > 3 else "LightGBM+競走得点"
    races = load(pred, method)
    tr = [r for r in races if r["date"] < cut]
    params = fit(tr)
    print(json.dumps({"method": method, "train_races": len(tr), "until": cut, "params": params}, ensure_ascii=False))
