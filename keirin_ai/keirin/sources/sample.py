"""練習用のダミーデータ（架空の選手・架空のレース）。ネット接続なしで全機能を試せる。

注意: ここで出る回収率は「架空の世界」での数字で、本物の競輪の成績とは無関係。
仕組み
- 選手ごとに隠れた「本当の強さ」があり、競走得点はそれにノイズを足したもの
- ライン（番手有利・先頭の逃げ）による有利不利がある
- 「市場（ほかのお客さん）」も強さをある程度見抜いてオッズを作る。控除率25%を引いた払戻
- 当日の調子など出走表に載らない情報があり、市場はそれをある程度知っているがAIは知らない
  （現実の競輪でも、公開データだけのAIはこの点で市場より不利）
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta

import numpy as np

from ..combos import BET_TYPES, combo_probs
from .base import OddsSnapshot, RaceCard, RaceResult, Source

VENUES = ["架空A", "架空B", "架空C"]
CLASSES = ["S1", "S2", "A1", "A2", "A3"]
TAKEOUT = 0.25


def _rng(*keys) -> np.random.Generator:
    h = hashlib.sha256("|".join(map(str, keys)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


class SampleSource(Source):
    name = "sample"

    def __init__(self, seed: int = 7, n_riders: int = 600, now: datetime | None = None,
                 market_skill: float = 1.0, market_noise: float = 0.2):
        self.seed = seed
        self.now = now
        self.market_skill = market_skill
        self.market_noise = market_noise
        rng = _rng(seed, "pool")
        self.ability = rng.normal(0, 1, n_riders)
        self.cls = np.array([CLASSES[min(4, max(0, int(2 - a * 1.3 + rng.normal(0, 0.5))))] for a in self.ability])
        self.style = rng.choice(["逃", "両", "追"], n_riders, p=[0.35, 0.2, 0.45])
        self.age = rng.integers(20, 50, n_riders)
        self.region = rng.integers(0, 6, n_riders)
        self._days: dict[str, dict] = {}

    # ---- 1日分の世界を作る（同じ日付なら毎回同じ内容） ----
    def _day(self, date: str) -> dict:
        if date in self._days:
            return self._days[date]
        rng = _rng(self.seed, date)
        races = []
        drift = 0.03 * _rng(self.seed, "drift", date[:7]).normal(0, 1, len(self.ability))
        for v_i, venue in enumerate(VENUES):
            used = set()
            for rno in range(1, 13):
                n = int(rng.choice([7, 9], p=[0.4, 0.6]))
                pool = [i for i in rng.permutation(len(self.ability)) if i not in used][:n]
                used.update(pool)
                races.append(self._race(rng, date, venue, v_i, rno, pool, drift))
        self._days[date] = {"races": races}
        return self._days[date]

    def _race(self, rng, date, venue, v_i, rno, pool, drift):
        race_id = f"{date.replace('-', '')}_{venue}_{rno:02d}"
        h, m = divmod(10 * 60 + 30 + 30 * (rno - 1) + 7 * v_i, 60)
        start = f"{h:02d}:{m:02d}"
        # ライン: 地区ごとにまとめ、最大3人
        lines: list[list[int]] = []
        for reg in sorted(set(self.region[pool])):
            members = [p for p in pool if self.region[p] == reg]
            members.sort(key=lambda p: (self.style[p] != "逃", -self.ability[p]))
            for k in range(0, len(members), 3):
                lines.append(members[k:k + 3])
        entries, true_u = [], []
        cars = list(rng.permutation(np.arange(1, len(pool) + 1)))
        for li, line in enumerate(lines):
            for pos, p in enumerate(line, start=1):
                car = int(cars.pop())
                ab = self.ability[p] + drift[p]
                score = round(float(95 + 6 * self.ability[p] + rng.normal(0, 2.0)), 2)
                bonus = 0.0
                if len(line) >= 2 and pos == 2:
                    bonus += 0.35  # 番手有利
                if len(line) >= 2 and pos == 1 and self.style[p] == "逃":
                    bonus += 0.15
                if len(line) == 1:
                    bonus -= 0.25  # 単騎不利
                home = int(rng.random() < 0.08)
                # 当日の調子など、出走表には載らないがお客さん（市場）はある程度つかんでいる情報
                condition = rng.normal(0, 0.4)
                true_u.append(1.1 * ab + bonus + 0.2 * home + condition)
                entries.append({
                    "race_id": race_id, "car_no": car, "rider_id": f"R{p:04d}", "name": f"選手{p:04d}",
                    "class_rank": self.cls[p], "score": score, "style": self.style[p],
                    "age": int(self.age[p]), "home": home, "line_id": li + 1, "line_pos": pos,
                    "line_size": len(line), "scratched": 0,
                })
        order = np.argsort([e["car_no"] for e in entries])
        entries = [entries[i] for i in order]
        true_u = np.array(true_u)[order]
        scratched = [entries[i]["car_no"] for i in range(len(entries)) if rng.random() < 0.01]

        def softmax(u):
            e = np.exp(u - u.max())
            return e / e.sum()

        active = [i for i, e in enumerate(entries) if e["car_no"] not in scratched]
        cars_a = [entries[i]["car_no"] for i in active]
        p_true = softmax(true_u[active])
        market_u = self.market_skill * true_u[active] + rng.normal(0, self.market_noise, len(active))
        p_mkt = softmax(market_u)
        # 結果（Plackett-Luce で1着から順に決める）
        remaining = list(range(len(active)))
        finish_order = []
        while remaining:
            w = p_true[remaining] / p_true[remaining].sum()
            pick = remaining[int(rng.choice(len(remaining), p=w))]
            finish_order.append(cars_a[pick])
            remaining.remove(pick)
        finishes = {c: i + 1 for i, c in enumerate(finish_order)}
        if rng.random() < 0.03 and len(finish_order) > 3:
            finishes[finish_order[-1]] = 9  # まれに失格・落車
        # オッズ（市場の見立て × 控除率）
        mk = combo_probs(cars_a, p_mkt)
        final_odds = {bt: {c: max(1.0, round((1 - TAKEOUT) / max(p, 1e-6), 1)) for c, p in mk[bt].items()}
                      for bt in BET_TYPES}
        pre_noise = _rng(self.seed, race_id, "pre")
        pre_odds = {bt: {c: max(1.0, round(o * float(np.exp(pre_noise.normal(0, 0.08))), 1)) for c, o in d.items()}
                    for bt, d in final_odds.items()}
        top = finish_order
        payouts = {
            "2車複": {"-".join(map(str, sorted(top[:2]))): 0},
            "2車単": {f"{top[0]}>{top[1]}": 0},
            "3連複": {"-".join(map(str, sorted(top[:3]))): 0},
        }
        for bt in payouts:
            for c in payouts[bt]:
                payouts[bt][c] = int(round(final_odds[bt][c] * 10)) * 10
        dt = datetime.strptime(f"{date} {start}", "%Y-%m-%d %H:%M")
        return {
            "race": {"race_id": race_id, "date": date, "venue": venue, "race_no": rno, "grade": "F1",
                     "start_time": start, "n_riders": len(entries)},
            "entries": entries, "scratched": scratched, "finishes": finishes, "payouts": payouts,
            "pre_odds": pre_odds, "final_odds": final_odds,
            "pre_ts": (dt - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S"),
            "final_ts": (dt + timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "start_dt": dt,
        }

    def _find(self, race: dict) -> dict:
        for r in self._day(race["date"])["races"]:
            if r["race"]["race_id"] == race["race_id"]:
                return r
        raise KeyError(race["race_id"])

    # ---- Source の実装 ----
    def fetch_cards(self, date: str) -> list[RaceCard]:
        return [RaceCard(r["race"], [dict(e) for e in r["entries"]]) for r in self._day(date)["races"]]

    def fetch_odds(self, race: dict) -> OddsSnapshot | None:
        r = self._find(race)
        return OddsSnapshot(race["race_id"], r["pre_odds"], r["pre_ts"], False)

    def fetch_result(self, race: dict) -> RaceResult | None:
        r = self._find(race)
        if self.now is not None and self.now < r["start_dt"] + timedelta(minutes=5):
            return None
        return RaceResult(
            race_id=race["race_id"], finishes=r["finishes"], scratched=r["scratched"], payouts=r["payouts"],
            final_odds=OddsSnapshot(race["race_id"], r["final_odds"], r["final_ts"], True),
        )
