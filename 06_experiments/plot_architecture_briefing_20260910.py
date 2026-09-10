"""Generate architecture and matched-pair briefing figures from saved evidence."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/intrmotiv_briefing_mpl')
import hashlib
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/architecture_briefing_20260910'
OUT.mkdir(parents=True, exist_ok=True)
font = Path(fm.findfont(fm.FontProperties(family='DejaVu Sans'), fallback_to_default=False))
if not font.is_file() or font.suffix.lower() not in ('.ttf', '.otf'):
    raise RuntimeError('A scalable font is required')
mpl.rcParams.update({'font.family':'DejaVu Sans', 'font.size':18,
    'axes.labelsize':18, 'axes.titlesize':19, 'xtick.labelsize':17,
    'ytick.labelsize':17, 'pdf.fonttype':42, 'svg.fonttype':'none'})

def save(fig, name):
    fig.savefig(OUT / (name+'.png'), dpi=180, facecolor='white')
    fig.savefig(OUT / (name+'.pdf'), facecolor='white')
    fig.savefig(OUT / (name+'_preview.png'), dpi=90, facecolor='white')
    plt.close(fig)

def box(ax, x,y,w,h,text,color='#edf3f8',size=18):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015',
        facecolor=color,edgecolor='#34495e',linewidth=1.4))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)

def arrow(ax,a,b,color='#34495e',style='-',rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=20,
        linewidth=1.8,color=color,linestyle=style,connectionstyle=f'arc3,rad={rad}'))

fig=plt.figure(figsize=(12,11))
ax=fig.add_axes([.03,.03,.94,.88]); ax.set(xlim=(0,12),ylim=(0,11)); ax.axis('off')
fig.suptitle('Implemented IntrMotiv families: signal and learning paths',fontsize=21,y=.98)
box(ax,.2,9.3,2.7,.95,'RGB + map ID\nfixed visual features')
box(ax,4.0,9.3,3.4,.95,'DG: learned projection\nBN → threshold; 16 units')
box(ax,8.6,9.3,3.1,.95,'Fixed CA3 register\n16 × 71 amplitudes')
arrow(ax,(2.92,9.77),(3.97,9.77)); arrow(ax,(7.42,9.77),(8.57,9.77))
ax.text(6,10.65,'Common backbone · R = 8, L = 64 · no learned recurrent matrix',ha='center',fontsize=18)
box(ax,.2,7.35,2.7,1.0,'Depth: 10 samples\n+ map ID bypass',color='#f2f2f2')
box(ax,4,7.35,3.4,1.0,'Controller + value head\nPPO; optional goal input')
box(ax,8.6,7.35,3.1,1.0,'Goal selector\nGraph or sampler',color='#f8f1e2')
arrow(ax,(2.92,7.85),(3.97,7.85)); arrow(ax,(8.57,7.85),(7.42,7.85))
arrow(ax,(10.15,9.27),(7.43,8.2))
arrow(ax,(5.7,8.38),(5.7,9.27),color='#009e73',style='--')
ax.text(5.95,8.82,'JOINT only: PPO → DG',fontsize=17,color='#007b59')
box(ax,.2,5.35,3.25,1.05,'Environment\naction → observation',color='#f2f2f2')
box(ax,4,5.35,3.4,1.05,'Worker reward\nflat gate / HIT / FIRST\n/ persistent goal',color='#f8f1e2',size=17)
box(ax,8.6,5.35,3.1,1.05,'DG temporal credit\n+ auxiliary losses',color='#eaf5ed',size=17)
arrow(ax,(4.0,7.45),(1.8,6.43)); arrow(ax,(3.48,5.88),(3.97,5.88))
arrow(ax,(5.7,6.43),(5.7,7.32),color='#009e73',style='--')
ax.plot([8.9,8.15,8.15],[9.27,8.92,5.87],color='#34495e',lw=1.8)
arrow(ax,(8.15,5.87),(8.57,5.87))
ax.plot([11.72,11.92,11.92,7.1],[5.87,5.87,10.4,10.4],color='#009e73',ls='--',lw=1.8)
arrow(ax,(7.1,10.4),(7.1,10.27),color='#009e73',style='--')
ax.text(.2,4.73,'Solid: information or recorded feedback   Dashed green: optimization path',fontsize=17)
box(ax,.2,2.8,5.5,1.45,'CPD contextual variant\nPast CA3 (± actions) gates DG logits\nDIRECT detaches; BPTT retains history\nTemporal candidate: no predictor',size=17)
box(ax,6.1,2.8,5.6,1.45,'W_REF variant\nSeparate frozen SCR detector defines arrival\nLive DG continues intrinsic learning\nSTOP / JOINT changes PPO → DG only',size=17)
box(ax,.2,.6,11.5,1.45,'Family differences\nSCR / SAT: ARR credit + C15 graph; PPO stops at DG\nDGP: HIT / FIRST × STOP / JOINT × LEG / FiLM\nPIC F_GATE: no goal, no graph · PIC G_SHARED / W_REF: goal, no graph',color='#f7f7f7',size=18)
save(fig,'architecture_paths')

p=ROOT/'06_experiments/results/persistent_intrinsic_control_submission_20260908/status_20260909.json'
d=json.loads(p.read_text())
rows=[]
for seed in (8,99,123):
    row={'seed':seed}
    for mode in ('STOP','JOINT'):
        name=f'PIC_W_REF_{mode}_S{seed}'
        coverage=next(x for x in d['coverage'] if x['id'].startswith('00_'+name+'_'))
        spatial=next(x for x in d['spatial'] if x['run_name']==name and x['target_env_steps']==200000000)
        row[mode]={'coverage':coverage['matched200'][0],
            'logged_window_samples':coverage['matched200'][1],
            'map_cosine':spatial['active_only_map_cosine'],
            'mono_fraction':spatial['mono_field_unit_fraction']}
    rows.append(row)
fig,axes=plt.subplots(1,2,figsize=(12,5.6))
colors=['#0072b2','#d55e00','#009e73']
for ax,key,title,ylims in zip(axes,['coverage','map_cosine'],
        ['Coverage at 180–200M','Map overlap at 200M'],[(65,95),(.25,.43)]):
    for row,color in zip(rows,colors):
        y=[row['STOP'][key],row['JOINT'][key]]
        ax.plot([0,1],y,'o-',color=color,lw=2,ms=8,label=f"Seed {row['seed']}")
    ax.set_xticks([0,1],['STOP','JOINT']);ax.set_xlim(-.25,1.25);ax.set_ylim(*ylims)
    ax.set_title(title,pad=15)
    ax.set_ylabel('Coverage AUC (visited cells)' if key=='coverage' else 'Active-only map cosine')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
axes[0].legend(frameon=False,fontsize=16,loc='upper left')
fig.suptitle('Frozen-reference control: paired training seeds',fontsize=21,y=.98)
fig.text(.5,.015,'Saved September 9 audit · n = 3 seeds · lines join paired runs; no inferential test',ha='center',fontsize=16)
fig.subplots_adjust(left=.11,right=.97,bottom=.18,top=.80,wspace=.42)
save(fig,'reference_gradient_pairs')
manifest={'source':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
    'font':str(font),'matplotlib':mpl.__version__,'pairs':rows,
    'selection':'All three W_REF STOP/JOINT seed pairs at the declared matched window',
    'transform':'No smoothing or statistical aggregation beyond the saved window means. Each line is one seed.',
    'architecture_source':'Sealed persistent_intrinsic_control_hotfix_20260908.tar.gz plus dated StudySpecs; diagram is schematic.'}
(OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(OUT)
