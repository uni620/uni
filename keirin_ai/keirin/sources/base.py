"""データ取得元の共通の形。

新しいサイト用の部品を作るときは、このクラスを継承して3つのメソッドを書くだけでよい。
AI・仮想購入・Excel 側は、どのサイトから取ったデータでも同じように動く。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RaceCard:
    race: dict                      # races テーブルの1行分（race_id, date, venue, race_no, grade, race_name, race_class, start_time, n_riders）
    entries: list[dict]             # entries テーブルの行（着順なし）


@dataclass
class OddsSnapshot:
    race_id: str
    odds: dict[str, dict[str, float]]   # {券種: {組み合わせ: 倍率}}
    fetched_at: str                     # "YYYY-MM-DD HH:MM:SS"
    is_final: bool


@dataclass
class RaceResult:
    race_id: str
    finishes: dict[int, int]                      # {車番: 着順}（失格等は 9）
    scratched: list[int] = field(default_factory=list)
    kimarite: dict[int, str] = field(default_factory=dict)        # {車番: 決まり手}（逃げ/捲り/差し/マーク）
    payouts: dict[str, dict[str, int]] = field(default_factory=dict)  # {券種: {組み合わせ: 100円あたり払戻}}
    final_odds: OddsSnapshot | None = None


class Source:
    name = "base"

    def fetch_cards(self, date: str) -> list[RaceCard]:
        """その日の全レースの出走表（予想並び＝ライン情報を含む）。"""
        raise NotImplementedError

    def fetch_odds(self, race: dict) -> OddsSnapshot | None:
        """発売中のオッズ。取れなければ None。"""
        return None

    def fetch_result(self, race: dict) -> RaceResult | None:
        """結果と払戻金。まだ確定していなければ None。"""
        raise NotImplementedError
