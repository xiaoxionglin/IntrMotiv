"""Checkpoint-only FiLM audit. No activation dominance inferred from weights."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'06_experiments/results/film_input_audit_20260914'
font=font_manager.findfont('DejaVu Sans',fallback_to_default=False)
assert Path(font).suffix.lower() in ('.ttf','.otf')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.titlesize':14,'pdf.fonttype':42})
runs=json.loads((OUT/'checkpoint_extract.json').read_text())
summary=[]; units=[];goals=[]
# Shared absolute limits make cross-run magnitude comparisons meaningful.
scale_lim=max(np.abs(np.array(r['weights']['decoder.target_modulation'])[:,:128]).max() for r in runs)
shift_lim=max(np.abs(np.array(r['weights']['decoder.target_modulation'])[:,128:]).max() for r in runs)
for r in runs:
 name=r['run'];w=r['weights'];W=np.array(w['decoder.state_layer.0.weight']);M=np.array(w['decoder.target_modulation']);g,b=np.split(M,2,1)
 assert W.shape==(128,1149) and M.shape==(16,256)
 cfg=r['config'];assert cfg['Hippo_n_feature']==16 and cfg['Hippo_L']+cfg['Hippo_R']-1==71
 c=np.mean(W[:,:1136]**2,1);d=np.mean(W[:,1136:1146]**2,1);ins=np.mean(W[:,1146:]**2,1)
 q=d/(c+d);raw=10*d/(1136*c+10*d);sg=g.std(0);sb=b.std(0)
 order=np.argsort(q); np.savez_compressed(OUT/(name+'.npz'),W=W,delta_scale=g,shift=b,unit_order=order,depth_normalized_weight_share=q)
 s=dict(run=name,checkpoint=r['checkpoint'],checkpoint_sha256=r['sha256'],scale_rms=float(np.sqrt(np.mean(g*g))),shift_rms=float(np.sqrt(np.mean(b*b))),goal_scale_sd_rms=float(np.sqrt(np.mean(sg*sg))),goal_shift_sd_rms=float(np.sqrt(np.mean(sb*sb))),median_dimension_normalized_depth_share=float(np.median(q)),median_total_weight_energy_depth_share=float(np.median(raw)),correlation_depth_scale=float(np.corrcoef(q,sg)[0,1]),correlation_depth_shift=float(np.corrcoef(q,sb)[0,1]),scale_goal_specific_energy_fraction=float(np.sum((g-g.mean(0))**2)/np.sum(g*g)),shift_goal_specific_energy_fraction=float(np.sum((b-b.mean(0))**2)/np.sum(b*b)),negative_gain_count=int(((1+g)<0).sum()))
 summary.append(s)
 for k in range(128):units.append(dict(run=name,unit=k,ca3_weight_rms=np.sqrt(c[k]),depth_weight_rms=np.sqrt(d[k]),instruction_weight_rms=np.sqrt(ins[k]),dimension_normalized_depth_share=q[k],total_energy_depth_share=raw[k],scale_sd_across_goals=sg[k],shift_sd_across_goals=sb[k]))
 for k in range(16):goals.append(dict(run=name,goal=k,scale_rms=np.sqrt(np.mean(g[k]**2)),shift_rms=np.sqrt(np.mean(b[k]**2)),centered_scale_rms=np.sqrt(np.mean((g[k]-g.mean(0))**2)),centered_shift_rms=np.sqrt(np.mean((b[k]-b.mean(0))**2))))
 fig=plt.figure(figsize=(12,11),layout='constrained'); gs=fig.add_gridspec(4,2,height_ratios=[1,1,.42,1])
 for row,(arr,lim,title) in enumerate([(g,scale_lim,'Goal-specific scale parameter Δγ (gain = 1 + Δγ)'),(b,shift_lim,'Goal-specific additive shift β')]):
  ax=fig.add_subplot(gs[row,:]);im=ax.imshow(arr[:,order],aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim,interpolation='nearest');ax.set_title(title);ax.set_ylabel('Goal ID');ax.set_yticks([0,3,7,11,15]);ax.set_xticks([]);fig.colorbar(im,ax=ax,pad=.01)
 ax=fig.add_subplot(gs[2,:]);ax.plot(np.arange(128),q[order],color='#0072B2');ax.axhline(.5,color='gray',ls='--');ax.set_ylim(0,1);ax.set_ylabel('Depth share');ax.set_xlabel('Hidden units sorted by dimension-normalized depth weight strength')
 for col,(v,label) in enumerate([(sg,'Scale SD across goals'),(sb,'Shift SD across goals')]):
  ax=fig.add_subplot(gs[3,col]);ax.scatter(q,v,s=18,alpha=.7,color='#0072B2');ax.set_xlabel('Dimension-normalized depth share');ax.set_ylabel(label);ax.set_xlim(0,1);ax.set_ylim(bottom=0);ax.set_title(f'Pearson r = {np.corrcoef(q,v)[0,1]:.2f}')
 fig.suptitle(name+'\n75.04M frames · 16 goals × 128 hidden units · weights only',fontsize=15)
 fig.savefig(OUT/(name+'.png'),dpi=150);fig.savefig(OUT/(name+'.pdf'));fig.savefig(OUT/(name+'_preview.png'),dpi=90);plt.close(fig)
pd.DataFrame(units).to_csv(OUT/'per_unit.csv',index=False);pd.DataFrame(goals).to_csv(OUT/'per_goal.csv',index=False)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(pd.DataFrame(summary)[['run','goal_scale_sd_rms','goal_shift_sd_rms','scale_goal_specific_energy_fraction','shift_goal_specific_energy_fraction','correlation_depth_scale','correlation_depth_shift']].to_string(index=False))
