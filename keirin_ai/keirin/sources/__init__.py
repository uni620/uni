from __future__ import annotations

from .base import OddsSnapshot, RaceCard, RaceResult, Source


def make_source(cfg) -> Source | None:
    """config の [source] name からデータ取得部品を作る。csv のときは None（import-csv で取り込む）。"""
    name = cfg.section("source").get("name", "sample")
    if name == "sample":
        from datetime import datetime

        from .sample import SampleSource
        return SampleSource(now=datetime.now())
    if name == "csv":
        return None
    raise ValueError(
        f"データ取得元 '{name}' はまだありません。サイト用の部品を keirin/sources/ に追加してください（README参照）。"
    )


__all__ = ["Source", "RaceCard", "OddsSnapshot", "RaceResult", "make_source"]
