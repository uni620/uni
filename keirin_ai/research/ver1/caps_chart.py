import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf'); plt.rcParams['font.family']='IPAPGothic'
cols=['race_id','date','bt','combo','odds','ret','ev']
A=pd.read_pickle('prereg3_T_all.pkl')[cols].assign(期間='8月上旬〜9/8')
B=pd.read_pickle('cross3_test_rev.pkl')[cols].assign(期間='9/9〜10/14')
C=pd.read_pickle('cross3_test.pkl')[cols].assign(期間='10/15〜11/19')
T=pd.concat([A,B,C]); NR=T.race_id.nunique(); print('全', NR, 'R', T.date.min(), T.date.max())
rng=np.random.default_rng(0)
def pick(mx,th):
    g=T[T.odds<=mx]; g=g.loc[g.groupby('race_id').ev.idxmax()]; return g[g.ev>=th]
rows=[]; cum={}
for th in [1.1,1.2]:
    for mx in [10,20,30]:
        h=pick(mx,th); v=h.ret.values.astype(float); idx=rng.integers(0,len(v),(2000,len(v))); b=v[idx].mean(1)
        rows.append(dict(EV=th,上限=mx,R=len(h),的中率=round((h.ret>0).mean()*100,1),回収率=round(v.mean(),1),下=round(np.percentile(b,2.5)),上=round(np.percentile(b,97.5)),収支=int(v.sum()-100*len(v)),
                         **{p:round(x.ret.mean(),1) for p,x in h.groupby('期間')}))
        d=h.groupby('date').ret.agg(['sum','size']); cum[(th,mx)]=(d['sum']-100*d['size']).cumsum()
tab=pd.DataFrame(rows); print(tab.to_string()); tab.to_csv('caps_table.csv',index=False)
C3={10:'#2a78d6',20:'#eb6834',30:'#1baf7a'}
fig,axes=plt.subplots(3,1,figsize=(7,13),dpi=150,gridspec_kw={'height_ratios':[1,1,1.1]})
for ax,th in zip(axes[:2],[1.1,1.2]):
    for mx in [10,20,30]:
        s=cum[(th,mx)]; x=pd.to_datetime(s.index)
        ax.plot(x,s.values,color=C3[mx],lw=2,label=f'{mx}倍以下')
        ax.annotate(f'{mx}倍 {int(s.values[-1]):+,}円',(x[-1],s.values[-1]),xytext=(4,0),textcoords='offset points',fontsize=8,color='#333',va='center')
    ax.axhline(0,color='#888',lw=1)
    for d in ['2026-09-09','2026-10-15']: ax.axvline(pd.Timestamp(d),color='#bbb',lw=1,ls='--')
    ax.set_title(f'累計の収支（1点100円）：期待値{th}以上を1レース1点',fontsize=11,loc='left')
    ax.set_ylabel('円'); ax.grid(axis='y',color='#eee'); ax.legend(fontsize=8,loc='upper left',frameon=False)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    ax.set_xlim(right=pd.Timestamp('2026-12-08'))
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%-m月'))
    ax.tick_params(labelsize=8)
ax=axes[2]; per=['8月上旬〜9/8','9/9〜10/14','10/15〜11/19']
lab=[];xs=[];i=0
w=0.13
for k,th in enumerate([1.1,1.2]):
    for j,mx in enumerate([10,20,30]):
        r=tab[(tab.EV==th)&(tab.上限==mx)].iloc[0]
        for p_i,p in enumerate(per):
            xpos=p_i+ (k*3+j-2.5)*w
            ax.bar(xpos,r[p],width=w-0.02,color=C3[mx],alpha=1.0 if th==1.1 else 0.5,label=(f'{mx}倍・EV{th}' if p_i==0 else None))
ax.axhline(100,color='#e34948',lw=1.2); ax.text(2.55,101,'100%',color='#e34948',fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels(per,fontsize=9); ax.set_ylabel('回収率 %')
ax.set_title('期間ごとの回収率（濃い色=EV1.1以上、薄い色=EV1.2以上）',fontsize=11,loc='left')
ax.legend(fontsize=7,ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(0.5,-0.08))
for sp in ['top','right']: ax.spines[sp].set_visible(False)
ax.grid(axis='y',color='#eee'); ax.set_ylim(0,max(160,tab[per].values.max()+10))
fig.text(0.01,0.005,'各期間とも、その期間を使わずに作ったモデルで計算（確定オッズ・30/20/10倍以下）。点線は期間の区切り。',fontsize=7,color='#555')
fig.tight_layout(rect=(0,0.015,1,1)); fig.savefig('caps_chart.png'); print('saved')
