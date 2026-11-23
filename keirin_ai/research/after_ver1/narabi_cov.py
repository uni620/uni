import hashlib,re,sqlite3,random
from pathlib import Path
c=sqlite3.connect("data/keirin.sqlite")
rids=[r[0] for r in c.execute("select race_id from races where has_result=1")]
random.seed(0); smp=random.sample(rids,600)
stat={"nofile":0,"none":0,"ok":0,"empty":0}; bym={}
for rid in smp:
    u=f"https://keirin.netkeiba.com/race/entry/?race_id={rid}"
    p=Path("data/cache")/(hashlib.sha1(u.encode()).hexdigest()+".html")
    if not p.exists(): stat["nofile"]+=1; continue
    h=p.read_text()
    m=re.search(r'section class="Contents_Box DeployYoso">(.*?)</section>',h,re.S)
    if not m: k="none"
    else:
        nums=re.findall(r'Shaban_Num">(\d)',m.group(1)); k="ok" if nums else "empty"
    stat[k]+=1; bym.setdefault(rid[:6],{}).setdefault(k,0); bym[rid[:6]][k]+=1
print(stat); print(dict(sorted(bym.items())))
