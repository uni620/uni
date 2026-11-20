"""楽天Kドリームスのオッズページから、2車単・2車複・3連複・ワイドのオッズを読む。

- オッズは「買うかどうか（期待値）」の判断と検証だけに使い、予想（AI）には入れない。
- 個人で使うだけ。取ったデータは GitHub などに置かない。
- ログインはしない。PoliteFetcher で 3 秒以上あけて取る。
- ページは 1 枚 約340KB と大きいので、保存するのはオッズ部分だけ（gzip）。

ページの番号: {場コード2桁}{開催初日 YYYYMMDD}{何日目 2桁}00{レース番号 2桁}
場コードは netkeirin の race_id の 9〜10 桁目と同じ。
"""
from __future__ import annotations

import gzip
import hashlib
import re
from datetime import date as Date
from pathlib import Path

from bs4 import BeautifulSoup

BASE = "https://keirin.kdreams.jp"

# 場コード → URL の名前。初めて使う場は、ページの題名に場名が入っているかで確かめる（_check）。
SLUGS = {
    "11": "hakodate", "12": "aomori", "13": "iwakitaira", "21": "yahiko", "22": "maebashi", "23": "toride",
    "24": "utsunomiya", "25": "omiya", "26": "seibuen", "27": "keiokaku", "28": "tachikawa", "31": "matsudo",
    "32": "chiba", "34": "kawasaki", "35": "hiratsuka", "36": "odawara", "37": "ito", "38": "shizuoka",
    "42": "nagoya", "43": "gifu", "44": "ogaki", "45": "toyohashi", "46": "toyama", "47": "matsusaka",
    "48": "yokkaichi", "51": "fukui", "53": "nara", "54": "mukomachi", "55": "wakayama", "56": "kishiwada",
    "61": "tamano", "62": "hiroshima", "63": "hofu", "71": "takamatsu", "73": "komatsushima", "74": "kochi",
    "75": "matsuyama", "81": "kokura", "83": "kurume", "84": "takeo", "85": "sasebo", "86": "beppu",
    "87": "kumamoto",
}

TYPES = {"2shatan": "2車単", "2shahuku": "2車複", "3renhuku": "3連複", "wide": "ワイド"}


def kd_id(race_id: str, meet_start: str) -> str:
    """netkeirin の race_id（YYYYMMDD + 場 + R）と開催初日 → Kドリームスの番号。"""
    d = Date.fromisoformat(f"{race_id[:4]}-{race_id[4:6]}-{race_id[6:8]}")
    day = (d - Date.fromisoformat(meet_start)).days + 1
    return f"{race_id[8:10]}{meet_start.replace('-', '')}{day:02d}00{race_id[10:12]}"


def odds_url(race_id: str, meet_start: str) -> str | None:
    slug = SLUGS.get(race_id[8:10])
    if slug is None:
        return None
    return f"{BASE}/{slug}/racedetail/{kd_id(race_id, meet_start)}/?pageType=odds&kakeshikiType=2shahuku"


def _num(s: str) -> float | None:
    s = s.replace(",", "").strip()
    m = re.match(r"^(\d+(?:\.\d+)?)", s)   # ワイドの「1.2～1.4」は低い方
    return float(m.group(1)) if m else None


def _key(bt: str, cars: list[int]) -> str:
    return f"{cars[0]}>{cars[1]}" if bt == "2車単" else "-".join(str(c) for c in sorted(cars))


def _pop_list(sec) -> dict[tuple[int, ...], float]:
    """人気順の表（上位50件まで）→ {(車番,...): オッズ}。"""
    out = {}
    for li in sec.select(".oddspop_table_wrapper li"):
        h = li.select_one(".header")
        if not h or "人気順" not in h.get_text():
            continue
        for sp in li.select("span.num"):
            o = sp.find_next_sibling("span", class_="odds")
            cars = tuple(int(x) for x in re.findall(r"\d+", sp.get_text()))
            v = _num(o.get_text()) if o else None
            if cars and v:
                out[cars] = v
    return out


def _exacta_matrix(sec) -> dict[tuple[int, int], float]:
    """2車単の表: 列の見出し＝1着、行の見出し＝2着。"""
    t = sec.select_one(".odds_table")
    out = {}
    if t is None:
        return out
    rows = t.select("tr")
    cols = [int(x.get_text(strip=True)) for x in rows[0].find_all("th") if x.get_text(strip=True).isdigit()]
    for tr in rows[1:]:
        ths = tr.find_all("th")
        if not ths or not ths[0].get_text(strip=True).isdigit():
            continue
        second = int(ths[0].get_text(strip=True))
        for first, td in zip(cols, tr.find_all("td")):
            v = _num(td.get_text())
            if v and first != second:
                out[(first, second)] = v
    return out


def _trio_tables(sec) -> dict[tuple[int, int, int], float]:
    out = {}
    for t in sec.select(".odds_table"):
        rows = t.select("tr")
        if not rows or not rows[0].find("th") or not rows[0].find("th").get_text(strip=True).isdigit():
            continue
        a = int(rows[0].find("th").get_text(strip=True))
        b = None
        for tr in rows[1:]:
            ths = [x.get_text(strip=True) for x in tr.find_all("th")]
            td = tr.find("td")
            if not td or "rider" in (td.get("class") or []):
                continue
            if len(ths) == 2:
                b, c = int(ths[0]), int(ths[1])
            elif len(ths) == 1 and b is not None:
                c = int(ths[0])
            else:
                continue
            v = _num(td.get_text())
            if v:
                out[(a, b, c)] = v
    return out


def parse_odds(html: str) -> dict:
    """→ {"odds": {券種: {組み合わせ: オッズ}}, "final": 確定か, "title": 題名, "as_of": "YYYY/MM/DD HH:MM"}"""
    s = BeautifulSoup(html, "html.parser")
    title = s.title.get_text(strip=True) if s.title else ""
    head = s.select_one(".odds_header")
    head_txt = head.get_text(" ", strip=True) if head else ""
    m = re.search(r"(\d{4}/\d{2}/\d{2} \d{2}:\d{2})現在", head_txt)
    odds: dict[str, dict[str, float]] = {}
    for k, bt in TYPES.items():
        sec = s.select_one(f"#JS_ODDSCONTENTS_{k}")
        if sec is None:
            continue
        pop = _pop_list(sec)
        if bt == "2車単":
            full = _exacta_matrix(sec)
            # 向きの確認: 人気順の表と食い違ったら、行と列を入れ替えて読む
            if pop and sum(full.get(c) == v for c, v in pop.items()) < len(pop) * 0.9:
                full = {(b, a): v for (a, b), v in full.items()}
            vals = {**full, **pop}
        elif bt == "3連複":
            vals = {**_trio_tables(sec), **pop}
        else:
            vals = pop
        if vals:
            odds[bt] = {_key(bt, list(c)): v for c, v in vals.items()}
    return {"odds": odds, "final": "確定オッズ" in head_txt, "title": title, "as_of": m.group(1) if m else None}


def check_page(info: dict, race: dict) -> bool:
    """取ったページが本当にそのレースか（題名の日付・場名・レース番号）を確かめる。"""
    t = info.get("title", "")
    d = race["date"]
    want_date = f"{int(d[:4])}年{int(d[5:7])}月{int(d[8:10])}日"
    return want_date in t and f" {int(race['race_no'])}R" in t and str(race.get("venue", "")) in t


def save_section(cache_dir: Path, url: str, html: str):
    """オッズ部分だけを gzip で保存（読み直し用）。"""
    s = BeautifulSoup(html, "html.parser")
    keep = [str(s.title or "")] + [str(x) for x in s.select(".odds_header")] + \
        [str(s.select_one(f"#JS_ODDSCONTENTS_{k}") or "") for k in TYPES]
    p = Path(cache_dir) / (hashlib.sha1(url.encode()).hexdigest() + ".html.gz")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(gzip.compress("\n".join(keep).encode("utf-8")))


def load_section(cache_dir: Path, url: str) -> str | None:
    p = Path(cache_dir) / (hashlib.sha1(url.encode()).hexdigest() + ".html.gz")
    return gzip.decompress(p.read_bytes()).decode("utf-8") if p.exists() else None
