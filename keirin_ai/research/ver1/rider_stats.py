"""選手ごとの「その競輪場での1着率・3着内率」と「位置別（先頭/番手/3番手以降/単騎）の1着率・3着内率・得意位置」。
すべてそのレースの日より前の成績だけで計算する。"""
import sqlite3, numpy as np, pandas as pd
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
c=sqlite3.connect(f'{S}/sim.sqlite')
E=pd.read_sql("select e.race_id,e.car_no,e.rider_id,e.finish,e.line_pos,e.line_size,e.scratched,r.date,r.venue,r.race_class from entries e join races r using(race_id)",c)
E=E[(E.scratched!=1)&~E.race_class.astype(str).str.contains('ガールズ')].copy()
E['finish']=pd.to_numeric(E.finish,errors='coerce')
size=E.line_size.fillna(1); pos=E.line_pos.fillna(1)
E['pos']=np.select([size<=1,pos==1,pos==2],['単騎','先頭','番手'],'3番手以降')
E['ok']=E.finish.notna().astype(float); E['w']=(E.finish==1).astype(float); E['t3']=(E.finish<=3).astype(float)
def before(keys):
    g=E.groupby(keys+['date'])[['ok','w','t3']].sum()
    cum=g.groupby(level=list(range(len(keys)))).cumsum()-g        # その日より前まで
    return E[keys+['date']].merge(cum.reset_index(),on=keys+['date'],how='left')[['ok','w','t3']].to_numpy()
allr=before(['rider_id']); ven=before(['rider_id','venue']); E[['n_all','w_all','t3_all']]=allr; E[['n_ven','w_ven','t3_ven']]=ven
POS=['先頭','番手','3番手以降','単騎']
for p in POS:
    sub=E.assign(ok=E.ok*(E.pos==p),w=E.w*(E.pos==p),t3=E.t3*(E.pos==p))
    g=sub.groupby(['rider_id','date'])[['ok','w','t3']].sum(); cum=g.groupby(level=0).cumsum()-g
    m=E[['rider_id','date']].merge(cum.reset_index(),on=['rider_id','date'],how='left')
    E[f'n_{p}'],E[f'w_{p}'],E[f't3_{p}']=m.ok.values,m.w.values,m.t3.values
K=5  # 少ない出走は本人の全体の率に寄せる
r_w=(E.w_all+K*0.15)/(E.n_all+K); r_t=(E.t3_all+K*0.40)/(E.n_all+K)
E['ven_win']=E.w_ven/E.n_ven.replace(0,np.nan); E['ven_top3']=E.t3_ven/E.n_ven.replace(0,np.nan)
E['ven_top3_s']=(E.t3_ven+K*r_t)/(E.n_ven+K); E['ven_win_s']=(E.w_ven+K*r_w)/(E.n_ven+K)
E['ven_top3_diff']=E.ven_top3_s-r_t
for p in POS:
    E[f'top3_{p}']=E[f't3_{p}']/E[f'n_{p}'].replace(0,np.nan); E[f'win_{p}']=E[f'w_{p}']/E[f'n_{p}'].replace(0,np.nan)
T3=np.column_stack([np.where(E[f'n_{p}']>=5,E[f'top3_{p}'],-1) for p in POS])
E['best_pos']=np.where(T3.max(1)>=0,np.array(POS)[T3.argmax(1)],'不明')
E['best_pos_top3']=np.where(T3.max(1)>=0,T3.max(1),np.nan)
E['at_best']=(E.pos==E.best_pos).astype(float).where(E.best_pos!='不明')
E['pos_today_n']=np.choose(np.array([POS.index(p) for p in E.pos]),[E[f'n_{p}'].values for p in POS])
E['pos_today_top3']=np.choose(np.array([POS.index(p) for p in E.pos]),[E[f'top3_{p}'].values for p in POS])
E['pos_today_win']=np.choose(np.array([POS.index(p) for p in E.pos]),[E[f'win_{p}'].values for p in POS])
E['pos_today_top3_s']=(np.nan_to_num(E.pos_today_top3*E.pos_today_n)+K*r_t)/(E.pos_today_n+K)
E['pos_gap_best']=E.pos_today_top3_s-((np.nan_to_num(E.best_pos_top3)*5+K*r_t)/(5+K)).where(E.best_pos!='不明')
cols=['race_id','car_no','rider_id','date','venue','pos','n_ven','ven_win','ven_top3','ven_win_s','ven_top3_s','ven_top3_diff','best_pos','best_pos_top3','at_best','pos_today_n','pos_today_win','pos_today_top3','pos_today_top3_s','pos_gap_best']
E[cols].to_pickle(f'{S}/rider_stats.pkl')
x=E[E.date>='2026-06-11']
print('6/11〜の出走',len(x),' その競輪場の出走が1回以上',round((x.n_ven>0).mean()*100,1),'%  平均',round(x.n_ven.mean(),1),'回')
print('得意位置が分かる',round((x.best_pos!='不明').mean()*100,1),'%  得意位置にいる',round(x.at_best.mean()*100,1),'%')
# 素の関係: 得意位置にいるかで3着内率は違うか
print(x.groupby('at_best').apply(lambda g: round((pd.to_numeric(E.loc[g.index,'finish'],errors='coerce')<=3).mean()*100,1)))
