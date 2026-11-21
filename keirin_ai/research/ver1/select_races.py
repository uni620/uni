"""勝負レースの絞り方: オッズ1点ルールの買い目に「得意位置・得意バンク」の条件を足すと回収率は変わるか。
条件は先に決めた6つだけ。学習+選択（〜10/10）で選び、最終確認（10/11〜）で1回だけ確かめる。"""
import pickle, numpy as np, pandas as pd, re
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
X=pickle.load(open(f'{S}/step5_scored.pkl','rb'))['オッズだけ']
RS=pd.read_pickle(f'{S}/rider_stats.pkl').set_index(['race_id','car_no'])
T=pd.read_pickle(f'{S}/step5_T.pkl'); ex=T[T.bt=='2車単'][['race_id','combo','q']].copy()
ex['first']=ex.combo.str.split('>').str[0].astype(int)
fav=ex.groupby(['race_id','first']).q.sum().reset_index().sort_values('q').drop_duplicates('race_id',keep='last').set_index('race_id')['first']
def feats(rid,combo):
    cars=[int(x) for x in re.findall(r'\d+',combo)]
    st=[RS.loc[(rid,c)] if (rid,c) in RS.index else None for c in cars]
    ab=[s.at_best for s in st if s is not None and pd.notna(s.at_best)]
    vd=[s.ven_top3_diff for s in st if s is not None and s.n_ven>0]
    f=fav.get(rid); fs=RS.loc[(rid,f)] if f is not None and (rid,f) in RS.index else None
    return (np.mean(ab) if ab else np.nan, np.mean(vd) if vd else np.nan, fs.at_best if fs is not None else np.nan)
rows=[]
for cap,th in ((10,1.1),(20,1.2),(30,1.2)):
    t=X[X.odds<=cap]; b=t.loc[t.groupby('race_id').ev.idxmax()]; b=b[b.ev>=th].copy()
    F=np.array([feats(r,c) for r,c in zip(b.race_id,b.combo)]); b['ab'],b['vd'],b['fav_ab']=F.T
    conds={'絞らない':b.ev>0,'F1 買い目の全員が得意位置':b.ab==1,'F2 買い目に得意位置でない選手がいる':b.ab<1,
           'F3 買い目の選手がこの競輪場を得意':b.vd>0,'F4 買い目の選手がこの競輪場を苦手':b.vd<0,
           'F5 1番人気が得意位置にいない':b.fav_ab==0,'F6 1番人気が得意位置':b.fav_ab==1}
    for name,m in conds.items():
        for per,pm in (('〜10/10',b.date<'2026-10-11'),('10/11〜',b.date>='2026-10-11')):
            v=b[m&pm].pay
            rows.append(dict(ルール=f'{cap}倍以下・期待値{th}',条件=name,期間=per,点=len(v),的中率=round((v>0).mean()*100,1) if len(v) else np.nan,回収率=round(v.mean(),1) if len(v) else np.nan))
R=pd.DataFrame(rows); R.to_csv(f'{S}/select_races.csv',index=False)
pd.set_option('display.width',250); print(R.pivot_table(index=['ルール','条件'],columns='期間',values=['点','回収率'],sort=False).to_string())
