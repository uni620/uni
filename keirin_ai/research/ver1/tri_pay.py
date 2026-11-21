"""保存済みの netkeirin 結果ページ（data/cache）から 3連単の払戻だけを取り出す。新しい通信はしない。"""
import sys, hashlib, sqlite3, re, unicodedata
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
sys.path.insert(0,'/home/user/uni/keirin_ai')
from keirin.sources.netkeirin import BASE
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
C=Path('/home/user/uni/keirin_ai/data/cache')
c=sqlite3.connect(f'{S}/sim.sqlite')
rids=[r[0] for r in c.execute("select race_id from races where has_result=1")]
rows=[]; miss=0
for rid in rids:
    p=C/(hashlib.sha1(f"{BASE}/race/result/?race_id={rid}".encode()).hexdigest()+'.html')
    if not p.exists(): miss+=1; continue
    s=BeautifulSoup(p.read_text(encoding='utf-8'),'html.parser')
    for tr in s.select('.Payout_Detail_Table tr.Tan3'):
        res,pay=tr.select_one('.Result'),tr.select_one('.Payout')
        if not(res and pay): continue
        nums=[int(x) for x in re.findall(r'\d+',unicodedata.normalize('NFKC',res.get_text(' ')))]
        pv=[int(x.replace(',','')) for x in re.findall(r'[\d,]+',unicodedata.normalize('NFKC',pay.get_text(' '))) if x.replace(',','')]
        for k in range(0,len(nums)-2,3):
            if k//3<len(pv): rows.append((rid,f"{nums[k]}>{nums[k+1]}>{nums[k+2]}",pv[k//3]))
T=pd.DataFrame(rows,columns=['race_id','combo','payout']); T.to_pickle(f'{S}/tri_pay.pkl')
print('レース',len(rids),'ページなし',miss,'3連単払戻',len(T),T.race_id.nunique()); print(T.head()); print(T.payout.describe())
