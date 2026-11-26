"""netkeiba の馬ごとのページを読む（個人の分析用）。
- 血統表 db.netkeiba.com/horse/ped/<id>/ → 5代先まで（62頭）。位置は F=父 M=母 の並び（例 "F"=父, "MF"=母の父, "FFM"=父の父の母）
- 馬のページ db.netkeiba.com/horse/<id>/ → 英名・性別・毛色・生年月日・調教師・馬主・生産者・産地・セリ価格・獲得賞金・通算成績・主な勝鞍・近親馬
"""
from __future__ import annotations

import re
import sqlite3

from bs4 import BeautifulSoup

GEN_BY_ROWSPAN = {"16": 1, "8": 2, "4": 3, "2": 4, None: 5}

SCHEMA = """
CREATE TABLE IF NOT EXISTS k_horse_jobs (horse_id TEXT PRIMARY KEY, ped_done INTEGER DEFAULT 0, ped_error TEXT,
  prof_done INTEGER DEFAULT 0, prof_error TEXT);
CREATE TABLE IF NOT EXISTS k_horse_ped (horse_id TEXT, pos TEXT, gen INTEGER, anc_id TEXT, anc_name TEXT,
  birth_year INTEGER, color TEXT, lineage TEXT, family_no TEXT, cell_text TEXT, PRIMARY KEY (horse_id, pos));
CREATE TABLE IF NOT EXISTS k_horses (horse_id TEXT PRIMARY KEY, name TEXT, name_en TEXT, sex TEXT, color TEXT,
  birth TEXT, trainer TEXT, trainer_id TEXT, owner TEXT, owner_id TEXT, breeder TEXT, origin TEXT, sale_price TEXT,
  prize_central TEXT, prize_local TEXT, record TEXT, main_wins TEXT, relatives TEXT, relative_ids TEXT, all_rows TEXT);
"""


def _t(x) -> str:
    return re.sub(r"\s+", " ", x.get_text(" ", strip=True)).strip() if x else ""


def _hid(href: str | None) -> str | None:
    m = re.search(r"/horse/([0-9a-zA-Z]+)/?$", href or "")
    return m.group(1) if m else None


def parse_ped(html: str, horse_id: str) -> list[dict]:
    s = BeautifulSoup(html, "html.parser")
    t = s.find("table", class_="blood_table")
    if not t:
        return []
    out, path, nth = [], [], {}
    for td in t.find_all("td"):
        g = GEN_BY_ROWSPAN.get(td.get("rowspan"), 5)
        if len(path) >= g:        # 同じ世代の2頭目は母、それ以外は父
            side = "M" if path[g - 1] == "F" and nth.get(g) == tuple(path[:g - 1]) else "F"
        else:
            side = "F"
        path = path[:g - 1] + [side]
        nth[g] = tuple(path[:g - 1])
        a = td.find("a")
        lines = [x.strip() for x in td.get_text("\n", strip=True).split("\n") if x.strip()]
        txt = " ".join(lines)
        my = re.search(r"\b(\d{4})\b\s*(\S*毛)?", txt)
        lin = next((x for x in lines if x.endswith("系")), None)
        fam = next((x for x in lines if re.fullmatch(r"\d{1,2}(?:-[a-z]+)?", x)), None)
        out.append({"horse_id": horse_id, "pos": "".join(path), "gen": g, "anc_id": _hid(a.get("href") if a else None),
                    "anc_name": _t(a) if a else (lines[0] if lines else None),
                    "birth_year": int(my.group(1)) if my else None, "color": my.group(2) if my else None,
                    "lineage": lin, "family_no": fam, "cell_text": txt})
    return out


def parse_profile(html: str, horse_id: str) -> dict | None:
    s = BeautifulSoup(html, "html.parser")
    p = s.find("table", class_="db_prof_table")
    title = s.select_one("div.horse_title")
    if not p and not title:
        return None
    d: dict = {"horse_id": horse_id}
    if title:
        d["name"] = _t(title.find("h1"))
        d["name_en"] = _t(title.find("p", class_="eng_name")) or None
        sc = _t(title.find("p", class_="txt_01")).replace("　", " ").split()
        d["sex"] = sc[0] if sc else None
        d["color"] = sc[1] if len(sc) > 1 else None
    rows = {}
    key = {"生年月日": "birth", "生産者": "breeder", "産地": "origin", "セリ取引価格": "sale_price",
           "獲得賞金 (中央)": "prize_central", "獲得賞金 (地方)": "prize_local", "通算成績": "record",
           "主な勝鞍": "main_wins", "近親馬": "relatives"}
    for tr in (p.find_all("tr") if p else []):
        th, td = tr.find("th"), tr.find("td")
        if not th or not td:
            continue
        k, v = _t(th), _t(td)
        rows[k] = v
        if k in key:
            d[key[k]] = v
        hrefs = [a.get("href", "") for a in td.find_all("a")]
        if k == "調教師":
            d["trainer"] = v
            d["trainer_id"] = next((m.group(1) for h in hrefs if (m := re.search(r"/trainer/(?:result/recent/)?(\w+)/", h))), None)
        elif k == "馬主":
            d["owner"] = v
            d["owner_id"] = next((m.group(1) for h in hrefs if (m := re.search(r"/owner/(?:result/recent/)?(\w+)/", h))), None)
        elif k == "近親馬":
            d["relative_ids"] = ",".join(i for h in hrefs if (i := _hid(h)))
    d["all_rows"] = "\t".join(f"{k}={v}" for k, v in rows.items())
    return d


def save_ped(con: sqlite3.Connection, rows: list[dict]) -> None:
    cols = ["horse_id", "pos", "gen", "anc_id", "anc_name", "birth_year", "color", "lineage", "family_no", "cell_text"]
    con.executemany(f"INSERT OR REPLACE INTO k_horse_ped ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                    [[r.get(c) for c in cols] for r in rows])


def save_profile(con: sqlite3.Connection, d: dict) -> None:
    cols = ["horse_id", "name", "name_en", "sex", "color", "birth", "trainer", "trainer_id", "owner", "owner_id",
            "breeder", "origin", "sale_price", "prize_central", "prize_local", "record", "main_wins", "relatives",
            "relative_ids", "all_rows"]
    con.execute(f"INSERT OR REPLACE INTO k_horses ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                [d.get(c) for c in cols])
