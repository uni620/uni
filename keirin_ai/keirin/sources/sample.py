"""練習用のダミーデータ（架空の選手・架空のレース）。ネット接続なしで全機能を試せる。

注意: ここで出る的中率・回収率は「架空の世界」での数字で、本物の競輪とは無関係。
     どの予想方法が良いかの判断には使えない（動作確認用）。

架空の世界のしくみ
- 選手ごとに隠れた「本当の強さ」と「自力で動く傾向」がある
- 3日間の大会があり、成績で 予選 → 準決勝 → 決勝 と勝ち上がる（負けた人は特選・一般へ）
- 位置の向き不向き（自力型の番手・追込型の先頭など）で有利不利が変わる
- 1着・2着には決まり手（逃げ/捲り/差し/マーク）がつく
- 当日の調子など、出走表に載らないがお客さん（市場）はある程度知っている情報がある
- オッズ = 市場の見立てから控除率25%を引いたもの
"""
from __future__ import annotations

import hashlib
from datetime import date as Date
from datetime import datetime, timedelta

import numpy as np

from ..combos import BET_TYPES, combo_probs
from .base import OddsSnapshot, RaceCard, RaceResult, Source

TAKEOUT = 0.25
VENUES = ["架空A", "架空B", "架空C"]
GRADES = [("F1", 0.55), ("F2", 0.2), ("G3", 0.2), ("G1", 0.05)]


def _rng(*keys) -> np.random.Generator:
    h = hashlib.sha256("|".join(map(str, keys)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _softmax(u):
    e = np.exp(u - u.max())
    return e / e.sum()


class SampleSource(Source):
    name = "sample"

    def __init__(self, seed: int = 7, n_riders: int = 900, now: datetime | None = None,
                 start: str = "2025-01-01", market_noise: float = 0.2):
        self.seed, self.now, self.market_noise = seed, now, market_noise
        self.start = Date.fromisoformat(start)
        rng = _rng(seed, "pool")
        self.n = n_riders
        self.ability = rng.normal(0, 1, n_riders)
        self.jiriki = np.clip(rng.beta(1.2, 1.2, n_riders), 0.02, 0.98)   # 自力で動く傾向
        self.region = rng.integers(0, 8, n_riders)
        self.age = rng.integers(20, 50, n_riders)
        # クラス: 強い順に S級 / A級 / A級チャレンジ
        order = np.argsort(-self.ability)
        self.cls = np.empty(n_riders, dtype=object)
        self.cls[order[: n_riders * 4 // 10]] = "S級"
        self.cls[order[n_riders * 4 // 10: n_riders * 8 // 10]] = "A級"
        self.cls[order[n_riders * 8 // 10:]] = "A級チャレンジ"
        self._cache: dict[str, list[dict]] = {}
        self._busy_until = np.full(n_riders, np.datetime64("1900-01-01"))
        self._made_until = self.start - timedelta(days=1)

    # ---------- 世界を日付順に作る（大会は3日間） ----------
    def _ensure(self, upto: Date):
        while self._made_until < upto:
            d = self._made_until + timedelta(days=1)
            for vi, venue in enumerate(VENUES):
                if (d - self.start).days % 4 == vi % 4 and (d - self.start).days % 4 != 3:
                    self._make_meet(venue, vi, d)
            self._made_until = d

    def _make_meet(self, venue: str, vi: int, d0: Date):
        rng = _rng(self.seed, venue, d0)
        grade = rng.choice([g for g, _ in GRADES], p=[p for _, p in GRADES])
        klass = "S級" if grade in ("G1", "G3") else rng.choice(["S級", "A級", "A級チャレンジ"], p=[0.4, 0.4, 0.2])
        size = 7 if klass == "A級チャレンジ" else 9
        n_races = 9
        free = np.where((self.cls == klass) & (self._busy_until < np.datetime64(d0)))[0]
        roster = rng.choice(free, size=min(len(free), size * n_races), replace=False)
        self._busy_until[roster] = np.datetime64(d0 + timedelta(days=3))
        points = {int(r): 0.0 for r in roster}
        seed = {int(r): self.ability[r] + rng.normal(0, 0.6) for r in roster}
        for day in range(3):
            dd = d0 + timedelta(days=day)
            if day == 0:
                ranked = sorted(roster, key=lambda r: -seed[int(r)])
            else:
                ranked = sorted(roster, key=lambda r: -(points[int(r)] + 0.3 * seed[int(r)]))
            chunks = [ranked[i:i + size] for i in range(0, len(ranked), size)]
            for k, members in enumerate(chunks):
                name = self._race_name(day, k, len(chunks))
                race = self._make_race(rng, dd, venue, vi, n_races - k, name, grade, klass, [int(m) for m in members])
                self._cache.setdefault(dd.isoformat(), []).append(race)
                for car, fin in race["finishes"].items():
                    rid = race["car_to_rider"][car]
                    points[rid] += max(0, 10 - fin) + (5 if name in ("準決勝", "初日特選") else 0)

    @staticmethod
    def _race_name(day: int, k: int, n: int) -> str:
        if day == 0:
            return "初日特選" if k == 0 else ("予選" if k < n - 1 else "一般")
        if day == 1:
            return "準決勝" if k < 3 else ("特選" if k < 5 else "一般")
        return "決勝" if k == 0 else ("特選" if k < 3 else ("特一般" if k < 5 else "一般"))

    def _make_race(self, rng, dd: Date, venue, vi, race_no, name, grade, klass, members):
        race_id = f"{dd.strftime('%Y%m%d')}_{venue}_{race_no:02d}"
        h, m = divmod(10 * 60 + 30 + 30 * (race_no - 1) + 7 * vi, 60)
        start = f"{h:02d}:{m:02d}"
        # ライン: 地区ごと、自力型が先頭、最大3人
        lines = []
        for reg in sorted({self.region[p] for p in members}):
            mem = sorted([p for p in members if self.region[p] == reg], key=lambda p: -self.jiriki[p])
            for i in range(0, len(mem), 3):
                lines.append(mem[i:i + 3])
        cars = list(rng.permutation(np.arange(1, len(members) + 1)))
        entries, true_u, car_to_rider = [], [], {}
        strength = {p: self.ability[p] for p in members}
        for li, line in enumerate(lines):
            for pos, p in enumerate(line, start=1):
                car = int(cars.pop())
                car_to_rider[car] = p
                u = 1.1 * self.ability[p]
                jr = self.jiriki[p]
                if len(line) == 1:
                    u -= 0.25                                   # 単騎は不利
                elif pos == 1:
                    u += 0.25 if jr > 0.5 else -0.35            # 追込型の先頭は不利
                elif pos == 2:
                    partner = line[0]
                    others = [strength[q] for q in members if q not in line]
                    strong_line = min(strength[partner], strength[p]) > (max(others) if others else -9)
                    if jr <= 0.5:
                        u += 0.35                               # 番手は有利
                    else:
                        u += 0.45 if strong_line else -0.15     # 自力型の番手: 強いラインなら逃げ切り、でなければ崩れやすい
                u += rng.normal(0, 0.4)                         # 当日の調子（出走表に載らない）
                true_u.append(u)
                entries.append({
                    "race_id": race_id, "car_no": car, "rider_id": f"R{p:04d}", "name": f"選手{p:04d}",
                    "class_rank": klass[:1] + ("1" if self.ability[p] > 0.8 else "2"),
                    "score": round(float(95 + 6 * self.ability[p] + rng.normal(0, 3.0)), 2),
                    "style": "逃" if jr > 0.6 else ("両" if jr > 0.4 else "追"),
                    "age": int(self.age[p]), "home": 0, "line_id": li + 1, "line_pos": pos,
                    "line_size": len(line), "scratched": 0,
                })
        order = np.argsort([e["car_no"] for e in entries])
        entries = [entries[i] for i in order]
        true_u = np.array(true_u)[order]
        cars_a = [e["car_no"] for e in entries]
        p_true = _softmax(true_u)
        p_mkt = _softmax(true_u + rng.normal(0, self.market_noise, len(true_u)))
        remaining, finish_order = list(range(len(cars_a))), []
        while remaining:
            w = p_true[remaining] / p_true[remaining].sum()
            pick = remaining[int(rng.choice(len(remaining), p=w))]
            finish_order.append(cars_a[pick])
            remaining.remove(pick)
        finishes = {c: i + 1 for i, c in enumerate(finish_order)}
        # 決まり手（1着・2着）
        pos_of = {e["car_no"]: (e["line_id"], e["line_pos"], e["line_size"]) for e in entries}
        kimarite = {}
        for rank, car in enumerate(finish_order[:2]):
            p = car_to_rider[car]
            lid, pos, size = pos_of[car]
            if rank == 1 and pos_of[finish_order[0]][0] == lid and pos == pos_of[finish_order[0]][1] + 1:
                kimarite[car] = "マーク"
            elif rng.random() < self.jiriki[p] * (1.2 if pos == 1 else 0.8):
                kimarite[car] = "逃げ" if (pos == 1 and size >= 2) else "捲り"
            else:
                kimarite[car] = "差し"
        mk = combo_probs(cars_a, p_mkt)
        final_odds = {bt: {c: max(1.0, round((1 - TAKEOUT) / max(pp, 1e-6), 1)) for c, pp in mk[bt].items()}
                      for bt in BET_TYPES}
        pre_r = _rng(self.seed, race_id, "pre")
        pre_odds = {bt: {c: max(1.0, round(o * float(np.exp(pre_r.normal(0, 0.08))), 1)) for c, o in dd_.items()}
                    for bt, dd_ in final_odds.items()}
        top = finish_order
        keys = {"2車複": "-".join(map(str, sorted(top[:2]))), "2車単": f"{top[0]}>{top[1]}",
                "3連複": "-".join(map(str, sorted(top[:3])))}
        payouts = {bt: {k: int(round(final_odds[bt][k] * 10)) * 10} for bt, k in keys.items()}
        dt = datetime.strptime(f"{dd.isoformat()} {start}", "%Y-%m-%d %H:%M")
        return {
            "race": {"race_id": race_id, "date": dd.isoformat(), "venue": venue, "race_no": race_no,
                     "grade": grade, "race_name": name, "race_class": klass, "start_time": start,
                     "n_riders": len(entries)},
            "entries": entries, "finishes": finishes, "kimarite": kimarite, "payouts": payouts,
            "pre_odds": pre_odds, "final_odds": final_odds, "car_to_rider": car_to_rider,
            "pre_ts": (dt - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S"),
            "final_ts": (dt + timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "start_dt": dt,
        }

    def _races(self, date: str) -> list[dict]:
        self._ensure(Date.fromisoformat(date))
        return sorted(self._cache.get(date, []), key=lambda r: (r["race"]["venue"], r["race"]["race_no"]))

    def _find(self, race: dict) -> dict:
        for r in self._races(race["date"]):
            if r["race"]["race_id"] == race["race_id"]:
                return r
        raise KeyError(race["race_id"])

    # ---------- Source の実装 ----------
    def fetch_cards(self, date: str) -> list[RaceCard]:
        return [RaceCard(dict(r["race"]), [dict(e) for e in r["entries"]]) for r in self._races(date)]

    def fetch_odds(self, race: dict) -> OddsSnapshot | None:
        r = self._find(race)
        return OddsSnapshot(race["race_id"], r["pre_odds"], r["pre_ts"], False)

    def fetch_result(self, race: dict) -> RaceResult | None:
        r = self._find(race)
        if self.now is not None and self.now < r["start_dt"] + timedelta(minutes=5):
            return None
        return RaceResult(race_id=race["race_id"], finishes=r["finishes"], kimarite=r["kimarite"],
                          payouts=r["payouts"],
                          final_odds=OddsSnapshot(race["race_id"], r["final_odds"], r["final_ts"], True))
