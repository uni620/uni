"""サイトに迷惑をかけないための取得ヘルパー（サイト用部品から使う）。

- robots.txt で禁止されているURLは取りに行かない
- 1回ごとに最低 interval 秒あける
- 同じURLは一度取ったらファイルに保存して再利用（確定済みの結果ページなど）
"""
from __future__ import annotations

import hashlib
import logging
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)


class PoliteFetcher:
    def __init__(self, user_agent: str, interval_sec: float = 3.0, cache_dir: str | Path = "data/cache"):
        self.ua = user_agent
        self.interval = max(1.0, float(interval_sec))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self._last = 0.0
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}

    def allowed(self, url: str) -> bool:
        u = urlparse(url)
        base = f"{u.scheme}://{u.netloc}"
        if base not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                self._wait()
                r = self.session.get(base + "/robots.txt", timeout=20)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except requests.RequestException:
                rp.parse([])
            self._robots[base] = rp
        return self._robots[base].can_fetch(self.ua, url)

    def _wait(self):
        dt = time.monotonic() - self._last
        if dt < self.interval:
            time.sleep(self.interval - dt)
        self._last = time.monotonic()

    def get(self, url: str, use_cache: bool = False, encoding: str | None = None) -> str:
        cache = self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".html")
        if use_cache and cache.exists():
            return cache.read_text(encoding="utf-8")
        if not self.allowed(url):
            raise PermissionError(f"robots.txt で禁止されているため取得しません: {url}")
        for attempt in range(3):
            self._wait()
            try:
                r = self.session.get(url, timeout=30)
                if r.status_code == 200:
                    if encoding:
                        r.encoding = encoding
                    text = r.text
                    if use_cache:
                        cache.write_text(text, encoding="utf-8")
                    return text
                if r.status_code in (403, 404):
                    raise RuntimeError(f"HTTP {r.status_code}: {url}")
            except requests.RequestException as e:
                log.warning("取得失敗(%s回目) %s: %s", attempt + 1, url, e)
            time.sleep(self.interval * (attempt + 2))
        raise RuntimeError(f"取得できませんでした: {url}")
