"""netkeirin（keirin.netkeiba.com）からデータを取る部品。

1日あたり : レース一覧 1ページ（その日の全レースID・場名・大会グレード）
1レースあたり: 出走表 / 並び予想 / 結果・払戻 の3ページ（ガールズは出走表だけで打ち切り）

マナーと利用規約（2026-11 時点で確認）
- robots.txt は全ページ許可。それでも 1回ごとに request_interval_sec 秒（初期値3秒）あける
- 確定したページはキャッシュに保存し、同じページを二度取りに行かない（途中で止めても続きから再開できる）
- 利用規約 第14条: 取ったデータは私的利用の範囲だけで使う。GitHub など公開の場所に置かない
- オッズは取得しない（HTMLに載っておらず、サイト内部の通信を解析する必要があるため）

ページの作りが変わると読み取れなくなる。そのときは parse_* 関数を直す。
"""
from __future__ import annotations

import logging
import re
import unicodedata
from datetime import date as Date

from bs4 import BeautifulSoup

from .base import RaceCard, RaceResult, Source
from .http import PoliteFetcher

log = logging.getLogger(__name__)

BASE = "https://keirin.netkeiba.com"
GRADE_MAP = {"GP": "GP", "GI": "G1", "GII": "G2", "GIII": "G3", "FI": "F1", "FII": "F2"}
BET_ROW_CLASS = {"Umaren": "2車複", "Umatan": "2車単", "Fuku3": "3連複", "Wide": "ワイド"}


def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").replace(" ", "").replace("　", "").strip()


def split_race_name(full: str) -> tuple[str | None, str]:
    """「Ｓ級 特選」→ ("S級", "特選")、「L級ガールズ予選1」→ ("ガールズ", "予選1")"""
    n = _nfkc(full)
    if n.startswith("L級") or "ガールズ" in n:
        return "ガールズ", n.replace("L級", "").replace("ガールズ", "")
    for prefix in ("A級チャレンジ", "S級", "A級"):
        if n.startswith(prefix):
            return prefix, n[len(prefix):]
    return None, n


def normalize_grade(g: str | None) -> str | None:
    if not g:
        return None
    g = _nfkc(g).upper()
    return GRADE_MAP.get(g, g)


def _int(s) -> int | None:
    m = re.search(r"-?\d+", str(s or "").replace(",", ""))
    return int(m.group()) if m else None


def _float(s) -> float | None:
    m = re.search(r"-?\d+(\.\d+)?", str(s or ""))
    return float(m.group()) if m else None


# ---------------- 読み取り（ページの作りに依存する部分） ----------------
def parse_race_list(html: str) -> tuple[list[str], dict[str, dict]]:
    """レース一覧 → (レースIDの一覧, {場コード: {"venue": 場名, "grade": グレード}})"""
    s = BeautifulSoup(html, "html.parser")
    ids = sorted(set(re.findall(r"race_id=(\d{12})", html)))
    venues = {}
    for li in s.select('li[id^="btn_jyoname_"]'):
        code = li["id"].replace("btn_jyoname_", "")
        a = li.find("a")
        name = a.find(string=True, recursive=False) if a else None
        grade = li.select_one(".Icon_Grade")
        venues[code] = {"venue": _nfkc(name or "") or None,
                        "grade": normalize_grade(grade.get_text(strip=True) if grade else None)}
    return ids, venues


def parse_entry(html: str, race_id: str) -> tuple[dict, list[dict]]:
    """出走表 → (レース情報, 選手の行)。並びは parse_narabi で別に取る。"""
    s = BeautifulSoup(html, "html.parser")
    title = s.title.get_text(strip=True) if s.title else ""
    full_name = s.select_one(".Race_Name")
    race_class, race_name = split_race_name(full_name.get_text(" ", strip=True) if full_name else "")
    data = s.select_one(".Race_Data")
    m = re.search(r"発走\s*(\d{1,2}:\d{2})", data.get_text(" ", strip=True) if data else "")
    grade = s.select_one(".Icon_GradeType")
    race = {
        "race_id": race_id,
        "date": f"{race_id[:4]}-{race_id[4:6]}-{race_id[6:8]}",
        "venue": title.split("競輪")[0].strip() or race_id[8:10],
        "race_no": int(race_id[10:12]),
        "grade": normalize_grade(grade.get_text(strip=True) if grade else None),
        "race_name": race_name or None,
        "race_class": race_class,
        "start_time": m.group(1).zfill(5) if m else None,
    }
    # バンク: 「周長距離は400メートル。見なし直線は56.9メートル。」（場によって書き方が少し違う）
    txt = s.get_text(" ", strip=True)
    b = re.search(r"周長距離[^。]*?(\d{3})\s*(?:メートル|m|ｍ)", txt)
    st = re.search(r"見なし直線[はもが]?\s*([\d.]+)\s*(?:メートル|m|ｍ)", txt)
    race["bank_len"] = float(b.group(1)) if b else None
    race["straight_len"] = float(st.group(1)) if st else None
    table = s.select_one("#RaceCard_Table_Fixed") or s
    heads = [th.get_text("", strip=True) for th in table.select("tr.Header th")]

    def col(label):
        for i, h in enumerate(heads):
            if h.startswith(label):
                return i
        return None

    ix = {k: col(k) for k in ["車", "競走得点", "脚質", "逃げ", "まくり", "差し", "マ"]}
    # S・H・B、着順の回数、勝率・連対率、ギヤ倍数（出走表に載るレース前の情報）
    exact = {h: i for i, h in enumerate(heads)}
    extra_cols = {"sb_s": "S", "sb_h": "H", "sb_b": "B", "n1": "1着", "n2": "2着", "n3": "3着", "nout": "着外",
                  "win_rate": "勝率", "ren2": "2連対率", "ren3": "3連対率", "gear": "ギヤ倍数"}
    rows = []
    for tr in table.select("tr.PlayerList"):
        tds = tr.find_all("td", recursive=False)
        a = tr.select_one('a[id^="names_"]')
        if not a or ix["車"] is None:
            continue
        rider_id = a["id"].split("_")[1]

        def cell(key, conv):
            i = ix.get(key)
            return conv(tds[i].get_text(" ", strip=True)) if i is not None and i < len(tds) else None

        style_cell = tds[ix["脚質"]].get_text(" ", strip=True) if ix["脚質"] is not None else ""
        style = next((c for c in style_cell if c in "逃両追"), None)
        rows.append({
            "race_id": race_id, "car_no": cell("車", _int), "rider_id": rider_id,
            "name": a.get_text(strip=True), "score": cell("競走得点", _float), "style": style,
            "km_nige": cell("逃げ", _int), "km_makuri": cell("まくり", _int),
            "km_sashi": cell("差し", _int), "km_mark": cell("マ", _int),
            "scratched": int("欠" in tr.get_text()),
        })
        for k, label in extra_cols.items():
            i = exact.get(label)
            v = _float(tds[i].get_text(" ", strip=True).replace("%", "")) if i is not None and i < len(tds) else None
            rows[-1][k] = v
    race["n_riders"] = len(rows)
    race.update(parse_market(s))
    return race, rows


def parse_market(s: BeautifulSoup) -> dict:
    """出走表の「3連単1番人気」表 → 1着の車番・オッズ・1着の支持率。無ければ空。"""
    t = s.select_one("table.OddsRateTable01")
    if t is None:
        return {}
    out = {}
    for tr in t.select("tr"):
        cells = [x.get_text(" ", strip=True) for x in tr.find_all(["th", "td"])]
        if not cells:
            continue
        if "1番人気" in cells[0]:
            m = re.search(r"着\s*(\d+)", cells[1]) if len(cells) > 1 else None
            out["mkt_fav1"] = int(m.group(1)) if m else None
            out["mkt_fav_odds"] = _float(cells[4].replace(",", "")) if len(cells) > 4 else None
        if cells[0] == "支持率" and len(cells) > 1 and cells[1].endswith("%"):
            out["mkt_sup1"] = _float(cells[1].rstrip("%"))
            out["mkt_sup1"] = out["mkt_sup1"] / 100 if out["mkt_sup1"] is not None else None
    return out


def parse_narabi(html: str) -> list[list[int]]:
    """並び予想 → [[先頭, 番手, 3番手], [単騎], ...]"""
    s = BeautifulSoup(html, "html.parser")
    lines, cur = [], []
    for box in s.select(".DeployBox .Shaban_InBox"):
        first = box.find("span")
        if first is None:
            continue
        cls = first.get("class") or []
        if "WakuSeparat" in cls:
            if cur:
                lines.append(cur)
            cur = []
        elif "Shaban_Num" in cls:
            n = _int(first.get_text())
            if n is not None:
                cur.append(n)
    if cur:
        lines.append(cur)
    return lines


def apply_narabi(rows: list[dict], lines: list[list[int]]):
    pos = {}
    for li, line in enumerate(lines, start=1):
        for p, car in enumerate(line, start=1):
            pos[car] = (li, p, len(line))
    next_id = len(lines) + 1
    for r in rows:
        if r["car_no"] in pos:
            r["line_id"], r["line_pos"], r["line_size"] = pos[r["car_no"]]
        else:  # 並び予想に載っていない選手は単騎扱い
            r["line_id"], r["line_pos"], r["line_size"] = next_id, 1, 1
            next_id += 1


def parse_result(html: str, race_id: str) -> RaceResult | None:
    s = BeautifulSoup(html, "html.parser")
    table = s.select_one("#All_Result_Table")
    if table is None:
        return None
    heads = [th.get_text("", strip=True) for th in table.select("tr.Header th")]
    i_car = heads.index("車番") if "車番" in heads else 2
    i_km = heads.index("決") if "決" in heads else None
    finishes, kimarite, scratched = {}, {}, []
    for tr in table.select("tr.PlayerList"):
        tds = tr.find_all("td", recursive=False)
        if len(tds) <= i_car:
            continue
        car = _int(tds[i_car].get_text())
        rank_txt = tds[0].get_text(strip=True)
        if car is None:
            continue
        if "欠" in rank_txt:
            scratched.append(car)
            continue
        rank = _int(rank_txt)
        finishes[car] = rank if rank is not None else 9   # 失格・落車・棄権など
        if i_km is not None and i_km < len(tds):
            k = tds[i_km].get_text(strip=True)
            if k:
                kimarite[car] = k
    if not finishes:
        return None
    payouts: dict[str, dict[str, int]] = {}
    for tr in s.select(".Payout_Detail_Table tr"):
        bt = next((BET_ROW_CLASS[c] for c in (tr.get("class") or []) if c in BET_ROW_CLASS), None)
        res, pay = tr.select_one(".Result"), tr.select_one(".Payout")
        if bt and res and pay:
            combo = _nfkc(res.get_text()).replace("=", "-")
            if bt != "2車単":
                combo = "-".join(str(x) for x in sorted(_int(p) for p in combo.split("-") if _int(p) is not None))
            v = _int(pay.get_text())
            if v is not None:
                payouts.setdefault(bt, {})[combo] = v
    return RaceResult(race_id=race_id, finishes=finishes, scratched=scratched, kimarite=kimarite, payouts=payouts)


# ---------------- Source ----------------
class NetkeirinSource(Source):
    name = "netkeirin"

    def __init__(self, fetcher: PoliteFetcher, skip_girls: bool = True):
        self.f = fetcher
        self.skip_girls = skip_girls
        self._venues: dict[str, dict] = {}

    def _past(self, race_date: str) -> bool:
        return race_date < Date.today().isoformat()

    def fetch_cards(self, date: str) -> list[RaceCard]:
        d = date.replace("-", "")

        def list_page(jyo: str) -> str:
            return self.f.get(f"{BASE}/race/api_race_list.html?pid=api_race_list&input=UTF-8&output=html"
                              f"&kaisai_date={d}&jyocd={jyo}&link_mode=", use_cache=self._past(date))

        ids, venues = parse_race_list(list_page(""))
        self._venues.update(venues)
        # 一覧は1つの場の分しか返さないので、残りの場も1ページずつ取る
        done = {rid[8:10] for rid in ids}
        all_ids = set(ids)
        for code in venues:
            if code not in done:
                all_ids.update(parse_race_list(list_page(code))[0])
        ids = sorted(all_ids)
        cards = []
        for rid in ids:
            if not rid.startswith(d):
                continue
            try:
                cards.append(self._card(rid))
            except Exception as e:  # 1レースの失敗で全体を止めない
                log.warning("出走表の取得に失敗 %s: %s", rid, e)
        return [c for c in cards if c is not None]

    def _card(self, rid: str) -> RaceCard | None:
        past = self._past(f"{rid[:4]}-{rid[4:6]}-{rid[6:8]}")
        race, rows = parse_entry(self.f.get(f"{BASE}/race/entry/?race_id={rid}", use_cache=past), rid)
        v = self._venues.get(rid[8:10], {})
        race["venue"] = v.get("venue") or race["venue"]
        race["grade"] = race["grade"] or v.get("grade")
        if race["race_class"] == "ガールズ" and self.skip_girls:
            return RaceCard(race, [])          # ガールズは対象外（レース情報だけ残す）
        lines = parse_narabi(self.f.get(f"{BASE}/race/api_get_narabi_yoso.html?input=UTF-8&output=html"
                                        f"&race_id={rid}", use_cache=past))
        apply_narabi(rows, lines)
        return RaceCard(race, rows)

    def fetch_result(self, race: dict) -> RaceResult | None:
        if race.get("race_class") == "ガールズ" and self.skip_girls:
            return None
        rid = race["race_id"]
        html = self.f.get(f"{BASE}/race/result/?race_id={rid}", use_cache=self._past(race["date"]))
        return parse_result(html, rid)
