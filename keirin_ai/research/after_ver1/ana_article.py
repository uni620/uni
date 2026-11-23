"""競輪戦線の穴狙い記事の主張を、手元の全レースで確かめる（ガールズ除く）。穴決着 = 2車単の払戻 5,000円以上（50倍以上）。"""
import sqlite3,pandas as pd,numpy as np
c=sqlite3.connect('/home/user/uni/keirin_ai/data/keirin.sqlite')
R=pd.read_sql("select race_id,date,venue,race_no,start_time,grade,race_class,bank_len,straight_len from races where has_result=1 and race_class not like '%ガールズ%'",c)
E=pd.read_sql("select race_id,car_no,score,line_id,line_pos,line_size,scratched,finish,sb_b,km_nige,km_makuri,km_sashi,km_mark from entries",c)
E=E[E.race_id.isin(R.race_id)]
P=pd.read_sql("select race_id,bet_type,combo,payout from payouts where bet_type in ('2車単','3連単')",c)
p2=P[P.bet_type=='2車単'].drop_duplicates('race_id').set_index('race_id').payout; p3=P[P.bet_type=='3連単'].drop_duplicates('race_id').set_index('race_id').payout
R=R[R.race_id.isin(p2.index)&R.race_id.isin(p3.index)].copy()
scr=set(E[E.scratched==1].race_id); R=R[~R.race_id.isin(scr)]
R['p2']=R.race_id.map(p2); R['p3']=R.race_id.map(p3); R['穴']=(R.p2>=5000); R['穴3']=(R.p3>=10000)
# 開催日目
R=R.sort_values(['venue','date']); d=pd.to_datetime(R.date)
days=R[['venue','date']].drop_duplicates().sort_values(['venue','date']); dd=pd.to_datetime(days.date)
days['meet']=(dd.groupby(days.venue).diff().dt.days.ne(1)).groupby(days.venue).cumsum()
days['n']=days.groupby(['venue','meet']).date.rank().astype(int); days['L']=days.groupby(['venue','meet']).date.transform('size')
days['日程']=np.where(days.n==1,'初日',np.where(days.n==days.L,'最終日','中日'))
R=R.merge(days[['venue','date','日程']],on=['venue','date'])
# 時間帯（その日の第1レースの発走時刻）
first=R.groupby(['venue','date']).start_time.min().rename('first'); R=R.join(first,on=['venue','date'])
R['時間帯']=pd.cut(R['first'].str[:2].astype(float)+R['first'].str[3:5].astype(float)/60,[0,9.5,12,17,24],labels=['モーニング','デイ','ナイター','ミッドナイト'])
g=R.grade.fillna('').str.upper(); R['グレード']=np.select([g.str.contains('G'),g=='F1'],['G3以上','F1'],'F2')
L=E.assign(lk=E.race_id+'_'+E.line_id.fillna(-E.car_no).astype(int).astype(str)).groupby('race_id').lk.nunique()
R['ライン数']=R.race_id.map(L); R['並び']=np.select([R.ライン数<=2,R.ライン数==3],['2分戦','3分戦'],'細切れ(4分以上)')
def show(col):
    t=R.groupby(col,observed=True).agg(R数=('穴','size'),穴割合_2車単50倍=('穴','mean'),穴割合_3連単100倍=('穴3','mean'),平均配当2車単=('p2','mean')); t.iloc[:,1:3]=(t.iloc[:,1:3]*100).round(1); t['平均配当2車単']=t.平均配当2車単.round(0); print(t.to_string(),'\n')
print('全',len(R),'R  穴(2車単50倍以上)',round(R.穴.mean()*100,1),'%')
for col in ('日程','時間帯','グレード','並び'): show(col)
show(['並び','時間帯'])
R.to_pickle('ana_R.pkl'); E.to_pickle('ana_E.pkl')
