"""netkeiba のページを読む（カレンダー・日別のレース一覧・レース結果）。中央競馬（場コード01〜10）だけを扱う。"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup

JRA_VENUES = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
              "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}


def _text(x) -> str:
    return re.sub(r"\s+", " ", x.get_text(" ", strip=True)).strip() if x else ""


def calendar_dates(html: str) -> list[str]:
    """race.netkeiba.com のカレンダー → 開催日 YYYYMMDD の一覧。"""
    return sorted(set(re.findall(r"kaisai_date=(\d{8})", html)))


def list_race_ids(html: str) -> list[str]:
    """db.netkeiba.com の日別一覧 → 中央競馬のレースID（12桁、5〜6桁目が01〜10）。"""
    ids = sorted(set(re.findall(r"/race/(\d{12})/", html)))
    return [i for i in ids if i[4:6] in JRA_VENUES]


def _id(href: str | None, kind: str) -> str | None:
    if not href:
        return None
    m = re.search(rf"/{kind}/(?:result/recent/)?([0-9a-zA-Z]+)/", href)
    return m.group(1) if m else None


def parse_race(html: str, race_id: str) -> dict:
    """db.netkeiba.com/race/<id>/ → {race, results, payouts, corners, laps}。読めるものは全部入れる。"""
    s = BeautifulSoup(html, "html.parser")
    race: dict = {"race_id": race_id, "venue_code": race_id[4:6], "venue": JRA_VENUES.get(race_id[4:6]),
                  "kai": int(race_id[6:8]), "nichi": int(race_id[8:10]), "race_no": int(race_id[10:12])}
    intro = s.select_one("div.data_intro")
    if intro:
        h1 = intro.find("h1")
        race["race_name"] = _text(h1)
        span = intro.find("diary_snap_cut") or intro.find("span")
        cond = _text(span)
        race["condition_text"] = cond
        m = re.search(r"(芝|ダ|障)(右|左|直線)?\s*(?:外|内|外-内|内-外|２周)?\s*(\d+)m", cond)
        if m:
            race["surface"], race["direction"], race["distance"] = m.group(1), m.group(2), int(m.group(3))
        for key, pat in (("weather", r"天候\s*:\s*(\S+)"), ("going", r"(?:芝|ダート)\s*:\s*(\S+)"), ("start_time", r"発走\s*:\s*(\d{1,2}:\d{2})")):
            m = re.search(pat, cond)
            race[key] = m.group(1) if m else None
        p = intro.find("p", class_="smalltxt")
        small = _text(p)
        race["detail_text"] = small
        m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", small)
        if m:
            race["date"] = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        m = re.search(r"日目\s*(.+?)\s*(\(|$)", small)
        race["class_text"] = m.group(1).strip() if m else None
    results = []
    tb = s.find("table", class_="race_table_01")
    if tb:
        head = [_text(th) for th in tb.find_all("tr")[0].find_all("th")]
        for tr in tb.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if not tds:
                continue
            cells = [_text(td) for td in tds]
            row = dict(zip(head, cells))
            links = {a.get("href", "") for a in tr.find_all("a")}
            get = lambda k: next((_id(h, k) for h in links if f"/{k}/" in h and _id(h, k)), None)  # noqa: E731
            w = row.get("馬体重", "")
            mw = re.match(r"(\d+)\(([-+]?\d+|前計不)\)", w)
            sexage = row.get("性齢", "")
            results.append({
                "race_id": race_id,
                "finish_text": row.get("着 順"), "waku": row.get("枠 番"), "umaban": row.get("馬 番"),
                "horse_name": row.get("馬名"), "horse_id": get("horse"),
                "sex": sexage[:1] or None, "age": sexage[1:] or None, "weight_carried": row.get("斤量"),
                "jockey": row.get("騎手"), "jockey_id": get("jockey"),
                "time": row.get("タイム"), "margin": row.get("着差"),
                "passing": row.get("通過"), "last3f": row.get("上り"),
                "win_odds": row.get("単勝"), "popularity": row.get("人 気"),
                "body_weight": int(mw.group(1)) if mw else None,
                "body_weight_diff": mw.group(2) if mw else None, "body_weight_text": w,
                "remarks": row.get("備考"), "trainer": row.get("調教師"), "trainer_id": get("trainer"),
                "owner": row.get("馬主"), "owner_id": get("owner"), "prize": row.get("賞金 (万円)"),
                "all_columns": "\t".join(f"{k}={v}" for k, v in row.items()),
            })
    payouts = []
    for pt in s.find_all("table", class_="pay_table_01"):
        for tr in pt.find_all("tr"):
            th = tr.find("th")
            tds = tr.find_all("td")
            if not th or len(tds) < 3:
                continue
            kind = _text(th)
            combos = [x for x in tds[0].get_text("\n", strip=True).split("\n") if x]
            pays = [x for x in tds[1].get_text("\n", strip=True).split("\n") if x]
            pops = [x for x in tds[2].get_text("\n", strip=True).split("\n") if x]
            for i, cb in enumerate(combos):
                payouts.append({"race_id": race_id, "bet_type": kind, "combo": cb.replace(" ", ""),
                                "payout": int(pays[i].replace(",", "")) if i < len(pays) and pays[i].replace(",", "").isdigit() else None,
                                "popularity": int(pops[i].replace(",", "")) if i < len(pops) and pops[i].replace(",", "").isdigit() else None})
    corners, laps = [], {}
    for t2 in s.find_all("table", class_="result_table_02"):
        summ = t2.get("summary", "")
        if "コーナー通過順位" in summ:
            for tr in t2.find_all("tr"):
                th, td = tr.find("th"), tr.find("td")
                if th and td:
                    corners.append({"race_id": race_id, "corner": _text(th), "order": _text(td)})
        elif "ラップタイム" in summ:
            for tr in t2.find_all("tr"):
                th, td = tr.find("th"), tr.find("td")
                if th and td:
                    laps[_text(th)] = _text(td)
    race["lap_text"] = laps.get("ラップ")
    race["pace_text"] = laps.get("ペース")
    race["n_horses"] = len(results)
    return {"race": race, "results": results, "payouts": payouts, "corners": corners}
