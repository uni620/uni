"""レースの格の点数（設計書「レースの格の点数表」）。

格 = レースの格（1〜6） + 大会グレードの補正（0〜1.0）
例: G1 の準決勝 = 5 + 0.8 = 5.8

レース名は大会ごとに違う（スタールビー賞など）ので、
- まず下の決まりで自動判定し、
- 判定できない名前は Excel の「格が不明なレース名」に一覧で出す。
- race_names.csv（列: race_name,kaku）に書けば、その点数が優先される。
"""
from __future__ import annotations

import csv
from pathlib import Path

GRADE_BONUS = {"GP": 1.0, "G1": 0.8, "G2": 0.6, "G3": 0.4, "F1": 0.2, "F2": 0.0}

# 上から順に調べる（「準決勝」は「決勝」を含むので先に調べる、など）
RULES: list[tuple[str, float]] = [
    ("準々決勝", 4),
    ("準決勝", 5),
    ("グランプリ", 6),
    ("決勝", 6),
    ("特別選抜予選", 5),
    ("特選予選", 5),
    ("初日特選", 5),
    ("ドリーム", 5),
    ("特別優秀", 4),
    ("二次予選", 4),
    ("特一般", 1),
    ("特選", 3),
    ("一次予選", 2),
    ("選抜", 2),
    ("予選", 2),
    ("一般", 1),
    ("賞", 5),          # スタールビー賞・ゴールデンレーサー賞など名前の付いた特別レース
]


def load_overrides(path: str | Path | None) -> dict[str, float]:
    if not path or not Path(path).exists():
        return {}
    out = {}
    for enc in ("utf-8-sig", "cp932"):
        try:
            with open(path, encoding=enc, newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("race_name") and row.get("kaku"):
                        out[row["race_name"].strip()] = float(row["kaku"])
            return out
        except UnicodeDecodeError:
            continue
    return out


def race_level(name: str | None, overrides: dict[str, float] | None = None) -> float | None:
    """レース名 → レースの格（1〜6）。分からなければ None。"""
    if not name:
        return None
    name = str(name).strip()
    if overrides and name in overrides:
        return overrides[name]
    for key, pts in RULES:
        if key in name:
            return float(pts)
    return None


def kaku(name: str | None, grade: str | None, overrides: dict[str, float] | None = None) -> float | None:
    lv = race_level(name, overrides)
    if lv is None:
        return None
    return lv + GRADE_BONUS.get(str(grade or "").upper(), 0.0)


def is_semifinal(name: str | None) -> bool:
    return bool(name) and "準決勝" in str(name) and "準々決勝" not in str(name)
