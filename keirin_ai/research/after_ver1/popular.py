"""保存済みの出走表ページから「3連単1番人気・オッズ・支持率」を読む（取得はしない）。"""
import sys, hashlib, os, re, sqlite3, pandas as pd
from bs4 import BeautifulSoup
BASE="https://keirin.netkeiba.com"; cache='/home/user/uni/keirin_ai/data/cache'
c=sqlite3.connect(sys.argv[1])
rows=[]
for rid, in c.execute("select race_id from races"):
    p=f"{cache}/{hashlib.sha1(f'{BASE}/race/entry/?race_id={rid}'.encode()).hexdigest()}.html"
    if not os.path.exists(p): continue
    s=BeautifulSoup(open(p,encoding='utf-8').read(),"html.parser")
    t=s.select_one("table.OddsRateTable01")
    if t is None: rows.append((rid,None,None,None,None,None,None,None,os.path.getmtime(p))); continue
    trs=[[x.get_text(" ",strip=True) for x in tr.find_all(['th','td'])] for tr in t.select("tr")]
    cars=[]; odds=None; sup=[None]*3
    for tr in trs:
        if tr and "1番人気" in tr[0]:
            cars=[int(m.group(1)) for x in tr[1:4] if (m:=re.search(r"着\s*(\d+)",x))]
            try: odds=float(tr[4].replace(",",""))
            except: pass
        if tr and tr[0]=="支持率":
            sup=[float(x.rstrip('%'))/100 if x.endswith('%') else None for x in tr[1:4]]
    rows.append((rid,*(cars+[None]*3)[:3],odds,*sup,os.path.getmtime(p)))
d=pd.DataFrame(rows,columns=['race_id','f1','f2','f3','fav_odds','sup1','sup2','sup3','mtime'])
d.to_pickle(sys.argv[2])
print(len(d),"レース中 情報あり",d.f1.notna().sum(), "支持率あり",d.sup1.notna().sum())
print(d.dropna().describe().round(3).to_string())
