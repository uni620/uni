"""頭AI用: コメントの特徴と、連係・対戦の成績（そのレースより前の日だけ）→ head3_feats.pkl"""
import sqlite3, re, numpy as np, pandas as pd
from collections import defaultdict
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite',timeout=60)
D=pd.read_sql("""select e.race_id,e.car_no,e.rider_id,e.name,e.line_id,e.line_pos,e.line_size,e.finish,e.scratched,r.date,i.comment
  from entries e join races r using(race_id) left join entry_info i using(race_id,car_no) where r.race_class not like '%ガールズ%'""",c)
D['finish']=pd.to_numeric(D.finish,errors='coerce'); D['comment']=D.comment.fillna('')
cm=D.comment
pat={'cm_自力':'自力|先行|逃げ|主導権|駆け|カマシ|かまし|捲り|まくり','cm_勝負':'勝負|先行|逃げ|主導権|駆け|積極|前々|攻め','cm_自在':'自在|何でも|決めず|臨機|流れ',
     'cm_単騎':'単騎|単独','cm_付く':'君|さん|へ|任せ|後ろ|番手|付いて|追走|マーク','cm_地区':'勢|コンビ|ライン|地元|同期|同県|連係',
     'cm_好調':'良い|いい|良く|上向|仕上が|手応え|好調|状態は|自信|戻って|動ける|軽い|調子','cm_不調':'悪|ダメ|駄目|不安|疲れ|落車|怪我|ケガ|痛|厳し|微妙|まだまだ|重い|イマイチ|いまいち'}
for k,p in pat.items(): D[k]=cm.str.contains(p).astype(float)
D['cm_なし']=(cm=='').astype(float); D['cm_長さ']=cm.str.len()
# 名指し: コメントに自分のラインの先頭（または前の選手）の苗字があるか / ライン外の選手の苗字があるか
D['sei']=D.name.fillna('').str.split(r'[\s　]').str[0].str[:3]
def nameflags(g):
    out=[]
    for r in g.itertuples():
        same=g[(g.line_id==r.line_id)&(g.car_no!=r.car_no)&g.line_id.notna()].sei.tolist() if pd.notna(r.line_id) else []
        oth=g[(g.line_id!=r.line_id)|g.line_id.isna()]; oth=oth[oth.car_no!=r.car_no].sei.tolist()
        c1=any(s and s[:2] in r.comment for s in same); c2=any(s and s[:2] in r.comment for s in oth)
        out.append((c1,c2))
    return pd.DataFrame(out,index=g.index,columns=['cm_味方名指し','cm_他線名指し'])
D=D.join(D.groupby('race_id',group_keys=False).apply(nameflags).astype(float))
# レース内: 自力・勝負と言っている人の数（先行争い）、自分以外に勝負宣言がいるか
D['R_自力宣言数']=D.groupby('race_id').cm_自力.transform('sum'); D['R_勝負宣言数']=D.groupby('race_id').cm_勝負.transform('sum')
D['lk']=D.race_id+'_'+D.line_id.fillna(-D.car_no).astype(int).astype(str)
ld=D[D.line_pos==1].set_index('lk')
for k in ('cm_自力','cm_勝負','cm_好調','cm_不調'): D['先頭の'+k]=D.lk.map(ld[k])
# 連係・対戦（日付順に、その日より前の結果だけ使う）
D=D.sort_values(['date','race_id','car_no']).reset_index(drop=True)
pair=defaultdict(lambda:[0,0,0])     # (先頭,番手) -> [一緒に走った回数, 先頭3着内, ワンツー(2人とも2着内)]
h2h=defaultdict(lambda:[0,0])        # (a,b) a<b -> [aがbに先着, 対戦数]
feat={}
for date,G in D.groupby('date',sort=True):
    for rid,g in G.groupby('race_id'):
        rows=g.to_dict('records'); by_lk=defaultdict(list)
        for r in rows: by_lk[r['lk']].append(r)
        for r in rows:
            mates=[m for m in by_lk[r['lk']] if m['car_no']!=r['car_no']]
            # 連係: 自分の前（番手なら先頭）か後ろとの過去
            pn=p3=p12=0
            for m in mates:
                if abs((m['line_pos'] or 0)-(r['line_pos'] or 0))==1:
                    a,b=(m['rider_id'],r['rider_id']) if (m['line_pos'] or 0)<(r['line_pos'] or 0) else (r['rider_id'],m['rider_id'])
                    v=pair[(a,b)]; pn+=v[0]; p3+=v[1]; p12+=v[2]
            # 対戦: 今回の相手全員への先着率
            w=n=0
            for o in rows:
                if o['car_no']==r['car_no']: continue
                a,b=r['rider_id'],o['rider_id']
                if a<b: v=h2h[(a,b)]; w+=v[0]; n+=v[1]
                else: v=h2h[(b,a)]; w+=v[1]-v[0]; n+=v[1]
            feat[(rid,r['car_no'])]=(pn,(p3+0.5)/(pn+1),(p12+0.2)/(pn+1),n,(w+1)/(n+2))
    # その日の結果で更新
    for rid,g in G.groupby('race_id'):
        g=g[g.finish.notna()&(g.scratched!=1)]
        rows=g.to_dict('records')
        for r in rows:
            for o in rows:
                if r['lk']==o['lk'] and (r['line_pos'] or 0)+1==(o['line_pos'] or 0):
                    v=pair[(r['rider_id'],o['rider_id'])]; v[0]+=1; v[1]+=r['finish']<=3; v[2]+=(r['finish']<=2 and o['finish']<=2)
                if r['rider_id']<o['rider_id']:
                    v=h2h[(r['rider_id'],o['rider_id'])]; v[0]+=r['finish']<o['finish']; v[1]+=1
F=pd.DataFrame([(k[0],k[1])+v for k,v in feat.items()],columns=['race_id','car_no','連係回数','連係時の先頭3着内率','連係時のワンツー率','対戦数','相手への先着率'])
D=D.merge(F,on=['race_id','car_no'],how='left')
D['先着率順位']=D.groupby('race_id').相手への先着率.rank(ascending=False)
# 自分の番手との連係（先頭目線）
D['番手との連係回数']=D.lk.map(D[D.line_pos==2].set_index('lk').連係回数).where(D.line_pos==1)
cols=[c for c in D.columns if c.startswith(('cm_','R_','先頭の'))]+['連係回数','連係時の先頭3着内率','連係時のワンツー率','対戦数','相手への先着率','先着率順位','番手との連係回数']
D[['race_id','car_no']+cols].to_pickle(f'{S}/head3_feats.pkl'); print(len(D),cols)
