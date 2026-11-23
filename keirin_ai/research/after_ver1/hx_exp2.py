from hx_lib import *
E,C=load(); PFC,CM,CC=C['PFC'],C['CM'],C['CC']; FEAT=[f for f in C['FEAT'] if f not in PFC]
E['dt']=pd.to_datetime(E.date)
E=E.sort_values(['rider_id','dt','race_id']).reset_index(drop=True)
g=E.groupby('rider_id')
def hist_mean(col,win=None):
    s=g[col].shift(1)
    return s.groupby(E.rider_id).transform(lambda v:v.expanding().mean() if win is None else v.rolling(win,min_periods=3).mean())
# (a) いつもと違う
E['普段の自力コメ']=hist_mean('cm_自力',20); E['自力コメの変化']=E.cm_自力-E.普段の自力コメ
E['普段の先頭率']=hist_mean('先頭',20); E['先頭の変化']=E.先頭-E.普段の先頭率
E['普段の番手率']=hist_mean('番手',20); E['番手の変化']=E.番手-E.普段の番手率
A=['普段の自力コメ','自力コメの変化','普段の先頭率','先頭の変化','普段の番手率','番手の変化']
# (b) 得点の動き
E['得点_前']=g.score.shift(1); E['得点_10走前']=g.score.shift(10)
E['得点の伸び(前走)']=E.score-E.得点_前; E['得点の伸び(10走)']=E.score-E.得点_10走前
B=['得点の伸び(前走)','得点の伸び(10走)']
# (e) 長期の成績
E['通算1着率']=hist_mean('win'); E['通算3着内率']=hist_mean('top3')
E['通算出走']=g.cumcount()
vg=E.groupby(['rider_id','venue'])
E['この場の1着率']=vg.win.shift(1).groupby([E.rider_id,E.venue]).transform(lambda v:(v.expanding().sum()+0.14*3)/(v.expanding().count()+3))
E['この場の出走']=vg.cumcount()
for k in ('通算1着率','通算3着内率','この場の1着率'): E[k+'_順位']=E.groupby('race_id')[k].rank(ascending=False)
EE=['通算1着率','通算3着内率','通算出走','この場の1着率','この場の出走','通算1着率_順位','通算3着内率_順位','この場の1着率_順位']
# (c) ライン
E['lkey']=E.race_id+'_'+E.line_id.fillna(-E.car_no).astype(int).astype(str)
L=E.groupby('lkey').agg(L得点=('score','mean'),L最高=('score','max'),LB率=('B率','sum'),L1着率=('通算1着率','sum'))
E=E.join(L,on='lkey')
for k in ('L得点','L最高','LB率','L1着率'):
    E[k+'_差']=E[k]-E.groupby('race_id')[k].transform('max')
ld=E[E.先頭==1].set_index('lkey').score; bn=E[E.番手==1].set_index('lkey').score
E['先頭−番手の得点']=E.lkey.map(ld)-E.lkey.map(bn)
CL=['L得点','L最高','LB率','L1着率','L得点_差','L最高_差','LB率_差','L1着率_差','先頭−番手の得点']
# (d) レースの種類
rn=E.race_name.fillna('')
E['種別']=np.select([rn.str.contains('決勝'),rn.str.contains('準決'),rn.str.contains('特選|特別|選抜|優秀|S級シャイニング'),rn.str.contains('予選'),rn.str.contains('一般|初日')],[5,4,3,2,1],0)
E['グレード']=E.grade.fillna('').map(lambda s:{'GP':6,'G1':5,'G2':4,'G3':3,'F1':2,'F2':1}.get(str(s).upper(),0))
E['周長']=pd.to_numeric(E.bank_len,errors='coerce'); E['レース番号']=E.race_no
E['場']=E.venue.astype('category').cat.codes
D=['種別','グレード','周長','レース番号','場']
E=E.sort_values(['date','race_id','car_no']).reset_index(drop=True)
E.to_pickle(f'{S}/hx2.pkl'); pickle.dump(dict(A=A,B=B,EE=EE,CL=CL,D=D),open(f'{S}/hx2_cols.pkl','wb'))
base=FEAT+CC
print('基準(+連係対戦)',evaluate(E,base),flush=True)
for nm,X in (('a いつもと違う',A),('b 得点の動き',B),('c ライン',CL),('d レースの種類',D),('e 長期・この場',EE)):
    print(nm,evaluate(E,base+X),flush=True)
print('全部',evaluate(E,base+A+B+CL+D+EE),flush=True)
