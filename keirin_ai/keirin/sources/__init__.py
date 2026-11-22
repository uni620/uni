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
    if name == "netkeirin":
        from .http import PoliteFetcher
        from .netkeirin import NetkeirinSource
        sc = cfg.section("source")
        cache = sc.get("cache_dir", "data/cache")
        fetcher = PoliteFetcher(sc.get("user_agent", "keirin-ai-personal/0.1"),
                                sc.get("request_interval_sec", 3.0),
                                cache if str(cache).startswith("/") else cfg.base / cache)
        return NetkeirinSource(fetcher, skip_girls=bool(sc.get("skip_girls", False)))
    raise ValueError(
        f"データ取得元 '{name}' はまだありません。サイト用の部品を keirin/sources/ に追加してください（README参照）。"
    )


__all__ = ["Source", "RaceCard", "OddsSnapshot", "RaceResult", "make_source"]
