"""netkeirin 本紙の予想（印と推奨買い目）を、保存済みの出走表ページから取り出す。新しい通信はしない。"""
import sys, hashlib, sqlite3, re, pickle
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
sys.path.insert(0,'/home/user/uni/keirin_ai')
from keirin.sources.netkeirin import BASE
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
C=Path('/home/user/uni/keirin_ai/data/cache')
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
races=pd.read_sql("select race_id,date from races where has_result=1",c)
MARK={'Icon_Honmei':'◎','Icon_Taikou':'○','Icon_Kurosan':'▲','Icon_Renka':'△','Icon_Hoshi':'×','Icon_Chui':'注'}
marks=[]; bets=[]; seen_cls=set(); miss=0
for rid in races.race_id:
    p=C/(hashlib.sha1(f"{BASE}/race/entry/?race_id={rid}".encode()).hexdigest()+'.html')
    if not p.exists(): miss+=1; continue
    s=BeautifulSoup(p.read_text(encoding='utf-8'),'html.parser')
    for tr in s.select('tr'):
        td=tr.select_one('td.Yoso01')
        if td is None: continue
        tds=tr.select('td')
        car=int(tds[1].get_text(strip=True)) if len(tds)>1 and tds[1].get_text(strip=True).isdigit() else None
        svg=td.select_one('svg'); m=None
        if svg is not None:
            for k in svg.get('class') or []:
                if k.startswith('Icon_') and k!='Icon_Shirushi01': seen_cls.add(k); m=MARK.get(k,k)
        cm=tr.select_one('td.Txt_L'); cm=cm.get_text(' ',strip=True) if cm else None
        marks.append((rid,car,m,td.get_text(strip=True),cm))
    for dl in s.select('.YosoKenkaiTxtBox .OddsWrap dl'):
        bt=dl.select_one('dt').get_text(strip=True)
        for dd in dl.select('dd'):
            num=dd.select_one('.Num'); 
            if num is None: continue
            main=[int(x.get_text(strip=True)) for x in num.find_all('span',class_='Wakuban',recursive=True) if x.find_parent(class_='Shaban_Other') is None]
            other=[int(x.get_text(strip=True)) for x in num.select('.Shaban_Other .Wakuban')]
            if bt=='3連単' and other: combos=[f"{main[0]}>{main[1]}>{o}" for o in other]
            elif bt=='3連単': combos=['>'.join(map(str,main))]
            elif bt=='2車単': combos=[f"{main[0]}>{main[1]}"]
            else: combos=['-'.join(map(str,sorted(main+other)))]
            for cb in combos: bets.append((rid,bt,cb))
M=pd.DataFrame(marks,columns=['race_id','car_no','mark','code','comment']); B=pd.DataFrame(bets,columns=['race_id','bet_type','combo'])
M.to_pickle(f'{S}/honshi_marks.pkl'); B.to_pickle(f'{S}/honshi_bets.pkl')
print('レース',len(races),'ページなし',miss,'印の行',len(M),'印の種類',M.mark.value_counts(dropna=False).to_dict(),'見つかったアイコン',seen_cls)
print('車番が取れない行',M.car_no.isna().sum()); print(B.bet_type.value_counts().to_dict(), B.race_id.nunique())
print(M.groupby(['mark','code']).size().head(20))
