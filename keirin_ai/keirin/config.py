"""設定ファイル（config.toml）の読み込み。"""
from __future__ import annotations

import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    raw: dict
    base: Path = field(default=ROOT)

    def path(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        if not p.is_absolute():
            p = self.base / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def section(self, name: str) -> dict:
        return self.raw.get(name, {})


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        path = ROOT / "config.toml"
        if not path.exists():
            shutil.copy(ROOT / "config.example.toml", path)
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return Config(raw=raw, base=path.resolve().parent)
