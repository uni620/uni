"""ネットで調べた条件ごとに、頭AIの本命の1着率（全体と、自信の帯の中）を比べる。4〜11月（各月その月より前で学習した頭AI）。"""
import pandas as pd,numpy as np
V=pd.read_pickle('hx_topp.pkl'); V=V[V.買い方=='2車単 3点'][['race_id','pr','headwin','ret']].rename(columns={'ret':'r2'})
D=pd.read_pickle('breakfeat.pkl'); AR=pd.read_pickle('ana_R.pkl').set_index('race_id')
E=pd.read_pickle('hx2.pkl')
# 頭（本命）の選手の情報: 頭AIの pr 最大 = hx_topp と同じ頭。hx_topp には car_no が無いので、同じモデルの頭を hx_conf / 再計算の代わりに「得点などで頭を特定」はできない → hx_verify と同じ月の頭を作り直す
H=pd.read_pickle('hx_topp_heads.pkl') if False else None
X=V.set_index('race_id').join(D).join(AR[['日程','時間帯','並び']],how='left')
X['自信帯']=pd.cut(X.pr,[0,.4,.5,.6,1],labels=['〜40%','40-50%','50-60%','60%〜'])
conds={
 '先行一車（先行候補が1人）':X.先行候補数==1,'先行候補が3人以上':X.先行候補数>=3,
 '2分戦':X.ライン数<=2,'細切れ(4分以上)':X.ライン数>=4,
 '得点1位と2位の差が5点以上':X['得点差1-2']>=5,'得点1位と2位の差が1点未満':X['得点差1-2']<1,
 '得点上位2人が同じライン':X.得点上位2人が同ライン==1,
 '強いラインの主導権確率55%超':X.強いラインのpB>0.55,
 '7車立て':X.頭数==7,'9車立て':X.頭数==9,
 '初日':X.日程=='初日','最終日':X.日程=='最終日',
 '333バンク':X.周長==333,'500バンク':X.周長==500,
 'モーニング':X.時間帯=='モーニング','ミッドナイト':X.時間帯=='ミッドナイト',
 'A級チャレンジ':X.級==0,'S級':X.級==2,'G3以上':X.グレード==2,
}
base=X.headwin.mean()
rows=[]
for k,m in conds.items():
    m=m.fillna(False); x=X[m]
    row={'条件':k,'R':len(x),'頭の1着率':round(x.headwin.mean()*100,1),'差':round((x.headwin.mean()-base)*100,1),'2車単3点的中':round((x.r2>0).mean()*100,1)}
    for b in ['50-60%','60%〜']:
        xb=x[x.自信帯==b]; ab=X[X.自信帯==b].headwin.mean()
        row[f'自信{b}の中で']=f"{xb.headwin.mean()*100:.0f}%({len(xb)}) 差{(xb.headwin.mean()-ab)*100:+.0f}" if len(xb)>=30 else '-'
    rows.append(row)
pd.set_option('display.width',250); print('全体の頭1着率',round(base*100,1),len(X))
print(pd.DataFrame(rows).sort_values('差',ascending=False).to_string(index=False))
