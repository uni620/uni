"""記事の考え方（ラインの総合力・先頭と番手の点数差・先行意欲・逃げイチ・近況と点数のずれ）を特徴にして、
LightGBM+競走得点 と比べる（月ごとの学び直し、6/11〜11/20）。"""
import sys, logging, time
sys.path.insert(0,'/home/user/uni/keirin_ai'); logging.basicConfig(level=logging.WARNING)
import numpy as np, pandas as pd
from keirin.config import load_config
from keirin.db import DB
from keirin import pipeline
from keirin.model import LGBModel
from keirin.features import FEATURES, SCORE_FEATURE, CARD_FEATURES
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
cfg=load_config('/home/user/uni/keirin_ai/config.toml'); db=DB(f'{S}/sim.sqlite')
f=pipeline.features_for(db,cfg); f=f[f.finish.notna()].copy()
f['score']=pd.to_numeric(f.score,errors='coerce')
f['lkey']=f.race_id+'_'+f.line_id.fillna(-f.car_no).astype(int).astype(str)
g=f.groupby('lkey')
f['line_mean']=g.score.transform('mean'); f['line_max']=g.score.transform('max')
f['line_size2']=g.car_no.transform('size')
lines=f.drop_duplicates('lkey')[['race_id','lkey','line_mean']]
lines['line_rank']=lines.groupby('race_id').line_mean.rank(ascending=False,method='min')
best=lines.groupby('race_id').line_mean.transform('max'); second=lines.groupby('race_id').line_mean.transform(lambda s:s.nlargest(2).iloc[-1] if len(s)>1 else s.iloc[0])
lines['line_gap_best']=np.where(lines.line_mean>=best,lines.line_mean-second,lines.line_mean-best)
f=f.merge(lines[['lkey','line_rank','line_gap_best']],on='lkey',how='left')
f['n_lines']=f.groupby('race_id').lkey.transform('nunique')
pos=f.line_pos.fillna(1)

lead=f[pos==1].set_index('lkey'); ban=f[pos==2].set_index('lkey')
f['leader_score']=f.lkey.map(lead.score); f['banete_score']=f.lkey.map(ban.score)
f['lf_gap']=f.leader_score-f.banete_score            # 先頭−番手（マイナス=番手のほうが格上）
f['me_minus_leader']=f.score-f.leader_score
kn=f.kimarite_n.replace(0,np.nan)
f['nige_rate']=(f.km_nige/kn) if 'km_nige' in f else np.nan
lead=f[pos==1].set_index('lkey')
f['leader_nige']=f.lkey.map(lead.nige_rate); f['leader_b']=f.lkey.map(lead.b_rate)
f['_senko']=((pos==1)&(f.nige_rate>=0.3)).astype(int)
f['n_senko']=f.groupby('race_id')._senko.transform('sum')
f['nige_ichi']=((f.n_senko==1)&(f.lkey.map(f[pos==1].set_index('lkey')._senko)==1)).astype(float)
r1=f.groupby('race_id').recent_finish.rank(method='average'); r2=f.groupby('race_id').score.rank(ascending=False,method='average')
f['form_vs_score']=r2-r1                               # プラス=点数の順位より近況が良い
ART=['line_mean','line_max','line_rank','line_gap_best','line_size2','n_lines','leader_score','banete_score','lf_gap','me_minus_leader','leader_nige','leader_b','n_senko','nige_ichi','form_vs_score']
VARS={'基準(+競走得点+出走表)':FEATURES+[SCORE_FEATURE]+CARD_FEATURES,'+記事の要素':FEATURES+[SCORE_FEATURE]+CARD_FEATURES+ART}
FOLDS=[("2026-06-11","2026-07-10"),("2026-07-11","2026-08-10"),("2026-08-11","2026-09-10"),("2026-09-11","2026-10-10"),("2026-10-11","2026-11-20")]
parts=[]; t0=time.time(); imp={}
for a,b in FOLDS:
    tr=f[f.date<a]; te=f[(f.date>=a)&(f.date<=b)]
    for name,cols in VARS.items():
        m=LGBModel(True); m.features=cols; m.name=name; m.fit(tr)
        parts.append(te[['race_id','date','car_no','finish','line_id','line_pos','line_size']].assign(p=m.predict(te)['p_win'].values,method=name))
        if b=='2026-11-20': imp[name]=pd.Series(m.boosters['win'].feature_importance('gain'),index=cols)
    print(a,f"{time.time()-t0:.0f}s",flush=True)
pd.concat(parts).to_pickle(f'{S}/wf_art.pkl')
i=imp['+記事の要素']; print((i/i.sum()).sort_values(ascending=False).head(20).round(3).to_string())
