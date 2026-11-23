"""Mr.Tさんの公開済み（レース後）の予想15件を取り出して整理する。手元での分析だけに使い、文章は外に出さない。"""
import sys,re,json; sys.path.insert(0,'/home/user/uni/keirin_ai')
from keirin.sources.http import PoliteFetcher
from bs4 import BeautifulSoup
import pandas as pd
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad/nk_yoso'
f=PoliteFetcher("keirin-ai-personal/0.1",3.0,S)
prof=open(S+'/mrt_profile.html').read()
ids=sorted(set(re.findall(r'detail/\?id=(b\d+_634)',prof)))
out=[]
for i in ids:
    h=f.get(f"https://keirin.netkeiba.com/yoso/detail/?id={i}",use_cache=True)
    s=BeautifulSoup(h,'html.parser')
    title=s.title.get_text(strip=True)
    m=re.search(r'】(\S+?)競輪 (.+?) (\d{4})年(\d+)月(\d+)日 (\d+)R (.+?) 競輪予想',title)
    txt=re.sub(r'\s+',' ',s.get_text(' ',strip=True))
    bets=[(int(a),int(b),int(c),int(y.replace(',',''))) for a,b,c,y in re.findall(r'3連単 通常 (\d) ＞ (\d) ＞ (\d) ([\d,]+)円',txt)]
    other=re.findall(r'(2車単|2車複|3連複|ワイド) 通常 ([\d ＞\-－=]+?) ([\d,]+)円',txt)
    pay=re.search(r'払い戻し金額 ([\d,]+)円',txt); tot=re.search(r'合計 ([\d,]+)円',txt)
    nar=re.search(r'進行方向 (.+?) 買い目',txt)
    res=re.search(r'着順 印 車番 選手名 人気 --> (.+?) 全着順',txt)
    mk=re.search(r'予想印 --> (.+?) 予想印を出走表に反映',txt)
    ken=re.search(r'見解 ウマい！ \d+ (.+?) この予想家をお気に入り',txt)
    out.append(dict(id=i,venue=m.group(1) if m else None,grade_name=m.group(2) if m else None,date=f"{m.group(3)}-{int(m.group(4)):02d}-{int(m.group(5)):02d}" if m else None,
        rno=int(m.group(6)) if m else None,race=m.group(7) if m else None,bets=bets,other=other,payout=pay.group(1) if pay else None,total=tot.group(1) if tot else None,
        narabi=nar.group(1) if nar else None,result=res.group(1)[:60] if res else None,marks=mk.group(1)[:300] if mk else None,view=ken.group(1) if ken else None))
pd.to_pickle(out,'/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad/mrt.pkl')
for o in out:
    print(o['date'],o['venue'],o['rno'],o['race'],'| 3連単',len(o['bets']),'点 他',len(o['other']),'| 合計',o['total'],'払戻',o['payout'])
