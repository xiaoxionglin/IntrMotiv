"""Render full-scan late-lift audits with explicit mean/median distinction."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT=Path(__file__).resolve().parent/'results/late_lift_audit_20260908'
font_manager.findfont('DejaVu Sans',fallback_to_default=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'axes.titlesize':17,
    'axes.labelsize':16,'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':14,
    'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
rows=[]; summaries=[]
for p in sorted(OUT.glob('*_full.csv.gz')):
    d=pd.read_csv(p).sort_values('train/env_steps'); d=d.dropna(subset=['lift'])
    if not len(d): continue
    name=p.name.removesuffix('_full.csv.gz'); end=d['train/env_steps'].max()
    d['bin']=np.floor(d['train/env_steps']/5e6)*5
    grouped=d.groupby('bin'); g=grouped.mean(numeric_only=True)
    med=grouped.lift.median()
    # A final single logging row just beyond 75M is not a 75–80M window.
    g=g.loc[g.index+5<=end/1e6]; med=med.reindex(g.index); x=g.index+2.5
    for b,q in grouped:
        row=dict(id=name,low_m=b,n=len(q),lift_mean=q.lift.mean(),lift_median=q.lift.median(),
            lift_max=q.lift.max(),fraction_gt2=(q.lift>2).mean(),fraction_gt100=(q.lift>100).mean())
        for c in ['success','target_rate','density','silence','entropy','edges','sensitivity','coverage','recruitment']:
            if c in q: row[c]=q[c].mean()
        if {'target_num','target_count','shuffle_num','shuffle_count'}<=set(q):
            a=q.target_num.sum()/q.target_count.sum(); b_rate=q.shuffle_num.sum()/q.shuffle_count.sum()
            row.update(target_activation_rate=a,shuffled_activation_rate=b_rate,pooled_lift=a/b_rate if b_rate else np.nan)
        rows.append(row)
    for label,lo,hi in [('early',.1*end,.5*end),('late',.5*end,end+1),('tail',end-10e6,end+1)]:
        q=d[d['train/env_steps'].between(lo,hi)]; l=q.lift
        row=dict(id=name,window=label,lo_m=lo/1e6,hi_m=hi/1e6,n=len(q),lift_mean=l.mean(),lift_median=l.median(),
            fraction_gt2=(l>2).mean(),fraction_gt100=(l>100).mean(),spike_share=l[l>100].sum()/l.sum() if l.sum() else np.nan)
        row.update({c:q[c].mean() for c in ['success','target_rate','density','silence','edges','sensitivity','coverage'] if c in q})
        summaries.append(row)
    fig,ax=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    ax=ax.ravel();ax[0].plot(x,g.lift,'o-',color='#0072B2',label='Mean of logged ratios')
    ax[0].plot(x,med,'s--',color='#D55E00',label='Median of logged ratios')
    if {'target_num','target_count','shuffle_num','shuffle_count'}<=set(d):
        totals=grouped[['target_num','target_count','shuffle_num','shuffle_count']].sum().reindex(g.index)
        pooled=(totals.target_num/totals.target_count)/(totals.shuffle_num/totals.shuffle_count)
        ax[0].plot(x,pooled,'^-',color='#009E73',label='Ratio of pooled rates')
    ax[0].axhline(1,color='0.5',ls=':');ax[0].set(ylabel='Target-hit lift',title='a  Lift: 5M-frame bins')
    if g.lift.max()>10:
        ax[0].set_yscale('symlog',linthresh=1)
        ax[0].set_ylim(0,g.lift.max()*1.5)
    ax[0].legend()
    ax[1].plot(x,100*g.success,'o-',color='#0072B2');ax[1].set(ylabel='Completed-option success (%)',title='b  Behavioral outcome',ylim=(0,None))
    ax[2].plot(x,g.coverage,'o-',color='#0072B2');ax[2].set(ylabel='Coverage AUC',title='c  Exploration',ylim=(0,100))
    ax[3].plot(x,100*g.density,'o-',color='#0072B2',label='DG density')
    ax[3].plot(x,100*g.edges,'s--',color='#D55E00',label='Known edges')
    ax[3].set(ylabel='Fraction (%)',title='d  Activity and graph support',ylim=(0,None));ax[3].legend()
    for a in ax: a.set_xlabel('Training frames (millions)'); a.grid(alpha=.15)
    # Split the timestamp off only for display; identities remain exact in CSV.
    title=name.split('_2026')[0].removeprefix('00_')
    fig.suptitle(title+'\nFull scan; complete 5M bins; means unless marked median',fontsize=18)
    fig.savefig(OUT/(name+'.png'),dpi=130);fig.savefig(OUT/(name+'.pdf'));plt.close(fig)
pd.DataFrame(rows).to_csv(OUT/'detail_5m_windows.csv',index=False)
pd.DataFrame(summaries).to_csv(OUT/'detail_summary.csv',index=False)
