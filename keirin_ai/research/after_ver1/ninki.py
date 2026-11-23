import hashlib,re,os,sqlite3,unicodedata
import pandas as pd
from bs4 import BeautifulSoup
BASE="https://keirin.netkeiba.com"
MAP={"Umaren":"2車複","Umatan":"2車単","Fuku3":"3連複"}
c=sqlite3.connect('data/keirin.sqlite')
races=c.execute("select race_id,date from races where has_result=1").fetchall()
rows=[];miss=0
for rid,d in races:
    p=f"data/cache/{hashlib.sha1(f'{BASE}/race/result/?race_id={rid}'.encode()).hexdigest()}.html"
    if not os.path.exists(p): miss+=1; continue
    s=BeautifulSoup(open(p,encoding='utf-8').read(),"html.parser")
    for tr in s.select(".Payout_Detail_Table tr"):
        bt=next((MAP[x] for x in (tr.get("class") or []) if x in MAP),None)
        if not bt: continue
        t=unicodedata.normalize("NFKC",tr.get_text(" ",strip=True))
        for pay,nk in re.findall(r"([\d,]+)円\s*(\d+)人気",t):
            rows.append((rid,d,bt,int(pay.replace(",","")),int(nk)))
df=pd.DataFrame(rows,columns="rid date bt pay nk".split())
print("races",len(races),"cache missing",miss)
def summ(x,label):
    out=[]
    for bt,g in x.groupby("bt"):
        n=g.rid.nunique(); h=g[g.nk==1]
        # 1番人気を100円ずつ買い続けた場合（同着・同率は払戻行ごと）
        out.append((label,bt,n,round(h.rid.nunique()/n*100,1),round(h.pay.sum()/n,1),round(g.groupby('rid').nk.min().le(3).mean()*100,1)))
    return out
r=summ(df,"全期間")+summ(df[(df.date>='2026-11-06')&(df.date<='2026-11-18')],"11/6-11/18")
print(pd.DataFrame(r,columns=["期間","券種","レース数","1番人気的中率%","1番人気回収率%","3番人気以内で決着%"]).to_string(index=False))

# AIの検証（11/6-11/18, eval2.sqlite の backtest）で当たった買い目は何番人気だったか
import numpy as np
e=sqlite3.connect(os.environ["EDB"])
b=pd.read_sql("select race_id rid,strategy,bet_type bt,status,payout from bets where mode='backtest'",e)
w=b[b.status=='won'].merge(df.groupby(['rid','bt']).agg(nk=('nk','min'),pay=('pay','max')).reset_index(),on=['rid','bt'])
w['帯']=pd.cut(w.nk,[0,1,3,10,999],labels=["1番人気","2-3番人気","4-10番人気","11番人気以下"])
print(pd.crosstab([w.strategy,w.bt],w['帯']).to_string())
print(w.groupby(['strategy','bt']).nk.median())
# 全レースの決着人気の分布（2車複・2車単・3連複）
df2=df.groupby(['rid','bt']).nk.min().reset_index()
df2['帯']=pd.cut(df2.nk,[0,1,3,10,999],labels=["1番人気","2-3番人気","4-10番人気","11番人気以下"])
print((pd.crosstab(df2.bt,df2['帯'],normalize='index')*100).round(1).to_string())
print(df2.groupby('bt').nk.median())
