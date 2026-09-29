"""Build two A0 alternatives combining transfer, prediction, and field structure.

The supplied layout is immutable. Retain the header, introduction, actual map
images, every trajectory vertex, and all peak markers. Edit only identified
section labels and bottom-left plot typography; compose the right column with
scoped imports using the existing A0 figure helpers. Existing pinned studies
and measurements remain authoritative; no telemetry or training is launched.
"""
from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

from compose_a0_poster_20260929 import (
    COLORS, FAMILY_COLORS, FAMILY_MARKERS, INK, PT_MM, ROOT, SCATTER, SVG,
    TABLE, dots, embed, np, pd, plt, sha, style, text, wrap_lines,
)
from render_poster_candidate_plots_20260929 import Candidate, Gallery, ROW_MM, paired, scalar_axes, verify_and_render
from hpc_runs.intrmotiv_study.version import WORKFLOW_VERSION
from matplotlib.lines import Line2D

SOURCE = Path('/home/xiaoxiong/.codex/attachments/56e5e916-3ac9-456f-9554-2f91aaab911e/Bernstein2026_IntrMotiv_layout.svg')
PREVIOUS = ROOT/'05_plans/poster_20260929/layout_versions'
DEFAULT_OUT = PREVIOUS/'revised'
FLAT = ROOT/'06_experiments/results/A0_poster_analysis_20260926/flat_goal_comparison'
TRANSFER = ROOT/'05_plans/poster_20260929/worker_random_transfer'
CONTINUOUS = ROOT/'05_plans/poster_20260929/continuous_fields'
LEFT_FIGURES = {
    'C01':('figure_1-9','figure_1-2','figure_1-1'),
    'C05':('figure_1-99','figure_1-3','figure_1-38'),
    'C15':('figure_1-26','figure_1-51','figure_1-95'),
}
LEFT_CAPTIONS = (
    'text131-6-1-1-7-5-2-3-3-1-1-6',
    'text131-6-1-1-7-5-2-3-3-1-1-6-2',
    'text131-6-1-1-7-5-2-3-3-1-1-6-2-5',
)
FAMILY_NAMES = {'CPD':'CA3 feedback','DGP':'DG policy','DGC':'Goal interfaces',
                'CPU cadence':'Controllers','Navigation8':'Algorithm screen','Source credit':'Source credit'}
VARIANTS = {
    'A_transfer_predictor_monofields':{
        'title':'Single-field fraction', 'recommended':True,
        'field_plot':'monofields_behavior',
        'field_caption':'More single fields ≠ better behaviour.',
        'field_protocol':'CA3 feedback · 27 variants × 3 seeds · DG 16 · 75M',
        'claim':'Single-field fraction has no positive exploration/control ordering in the 81-run CA3-feedback family.',
        'risk':'Online coverage is near its ceiling. Maps use a recent behavior window; target hits are historical internal events. '
               'Architectural association does not establish that monofields have no causal benefit.',
    },
    'B_transfer_predictor_dominance':{
        'title':'Continuous field dominance', 'recommended':False,
        'field_plot':'dominance_behavior',
        'field_caption':'Field dominance ≠ better behaviour.',
        'field_protocol':'CA3 feedback · 4 variants × 3 seeds · DG 16 · 75M',
        'claim':'Continuous dominance is negatively associated with online coverage and weakly associated with target hits in the locally available CPD subset.',
        'risk':'Only four CPD variants have local unit-level continuous scores. Minimum-threshold dominance is distinct '
               'from spatial concentration and from the binary field fraction. Protocol dependence limits generalization.',
    },
}


def find(root, ident):
    return next(e for e in root.iter() if e.get('id') == ident)


def remove(root, element):
    parent = next(p for p in root.iter() if element in list(p))
    parent.remove(element)


def query_bounds(path: Path):
    result = subprocess.run(['inkscape',str(path),'--query-all'],capture_output=True,text=True,check=True)
    return {r[0]:np.array([float(v)*25.4/96 for v in r[1:]])
            for r in csv.reader(result.stdout.splitlines()) if len(r)==5}


def matrix(group):
    numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',group.get('transform',''))
    if len(numbers) != 6 or not group.get('transform','').startswith('matrix('):
        raise ValueError('Expected audited six-component figure transform')
    a,b,c,d,x,y = map(float,numbers)
    if abs(b)+abs(c) > 1e-6 or not np.isclose(a,d,rtol=1e-5):
        raise ValueError('Bottom-left figures must use a uniform axis-aligned transform')
    return a,x,y


def strip_text(group):
    for p in list(group.iter()):
        for c in list(p):
            if c.tag == f'{{{SVG}}}text':
                p.remove(c)


def preserved_data(group):
    """Fingerprint the actual rasters, trajectory vertices and peak markers.

    Axis/legend primitives are annotations. Child-group translation changes
    placement but leaves stored data geometry untouched.
    """
    records = []
    def walk(e, annotation=False):
        annotation |= e.get('id','').startswith(('matplotlib.axis','legend_'))
        if not annotation and e.tag == f'{{{SVG}}}image':
            records.append(('image',tuple(sorted((k,v) for k,v in e.attrib.items() if k.endswith('href')))))
        if not annotation and e.get('id','').startswith(('line2d_','PathCollection_')):
            records.append(('data',e.get('id'),tuple((z.tag,tuple(sorted(z.attrib.items())))
                for z in e.iter() if z.tag in {f'{{{SVG}}}path',f'{{{SVG}}}use'})))
        for child in e:
            walk(child,annotation)
    walk(group)
    return hashlib.sha256(repr(records).encode()).hexdigest()


def peak_label_positions(labels, map_x, map_y, width, height):
    """Place selected IDs near their peaks with readable physical separation.

    Peak data do not move. Four labels fit in the map, and short leaders expose
    any displaced association. The final Inkscape check uses actual glyph bounds.
    """
    placed=[]
    for (bin_x,bin_y),shared in labels.groupby(['peak_x_bin','peak_y_bin']):
        message=','.join(map(str,sorted(shared.unit.astype(int))))
        px=map_x+(bin_x+.5)/19*width
        py=map_y+height-(bin_y+.5)/19*height
        label_width=len(message)*6.7+2
        candidates=[]
        for dx in (-20,-12,-5,2,10,18):
            for dy in (-20,-12,-2,10,18,26):
                x=min(max(px+dx,map_x+2),map_x+width-label_width-2)
                y=min(max(py+dy,map_y+12),map_y+height-3)
                box=np.array([x-1,y-10,label_width+2,12.])
                if any((np.minimum(box[:2]+box[2:],prior['box'][:2]+prior['box'][2:])-
                        np.maximum(box[:2],prior['box'][:2])>-.5).all() for prior in placed):
                    continue
                candidates.append(((x+label_width/2-px)**2+(y-4-py)**2,x,y,box))
        if not candidates:raise ValueError('No collision-free placement for selected peak labels')
        _,x,y,box=min(candidates,key=lambda r:r[0])
        placed.append({'unit':int(shared.unit.min()),'message':message,'x':x,'y':y,
                       'point_x':px,'point_y':py,'box':box,'width':label_width})
    return placed


def section(layer, number, title, x, baseline):
    ET.SubElement(layer,f'{{{SVG}}}rect',{'id':f'section-{number}-badge','x':str(x),
        'y':str(baseline-17),'width':'22','height':'22','rx':'3','fill':'#253B57'})
    text(layer,f'section-{number}-number',x+5,baseline,[str(number)],48,bold=True,color='white')
    text(layer,f'section-{number}-title',x+31,baseline,[title],48,bold=True)


def paragraph(layer, ident, x, baseline, message, width=383, size=40, bold=False, step=18):
    lines = wrap_lines(message,width,size,bold)
    text(layer,ident,x,baseline,lines,size,step,bold)
    return baseline+(len(lines)-1)*step


def left_typography(root, bounds, g: Gallery):
    """Keep all plotted data; replace tiny labels in page coordinates."""
    old = find(root,'poster-right-20260929')
    old.remove(find(old,'right-exploration-heading-0'))
    for ident in LEFT_CAPTIONS:
        remove(root,find(root,ident))
    layer = ET.SubElement(root,f'{{{SVG}}}g',{'id':'revised-left-typography',
        f'{{{INK}}}groupmode':'layer',f'{{{INK}}}label':'1 · Intrinsic designs · readable labels'})
    section(layer,1,'Three intrinsic designs',19.5,741)
    text(layer,'left-common-protocol',29,759,
         ['Seed 99 · map labels: unit : max · ○ starts; ★ DG target events'],30)
    scales = g.read(FLAT/'place_field_individual_scales.csv')
    peaks = g.read(FLAT/'all_active_dg_peak_locations.csv')
    audit = []
    for index,(condition,fig_ids) in enumerate(LEFT_FIGURES.items()):
        subtitle_y, map_y = ((780,801),(923,944),(1062,1083))[index]
        subtitle = {'C01':'1a  C01 · Non-goal-conditioned',
                    'C05':'1b  C05 · Uniform goal choice',
                    'C15':'1c  C15 · Frontier goal choice'}[condition]
        text(layer,condition+'-subheading',29,subtitle_y,[subtitle],36,bold=True,color=COLORS[condition])
        selected = scales[(scales.condition==condition)&(scales.seed==99)].sort_values('rank')
        fields = find(root,fig_ids[0]); prior = preserved_data(fields)
        scale,_,_ = matrix(fields)
        axes = [e for e in fields if e.get('id','').startswith('axes_')]
        if len(axes)!=8 or len(selected)!=4:
            raise ValueError('Expected four maps and four separate color scales')
        strip_text(fields)
        for n,(_,row) in enumerate(selected.iterrows()):
            image = next(e for e in axes[n].iter() if e.tag == f'{{{SVG}}}image')
            x0,y0,w,h = bounds[image.get('id')]
            x,y = (44 if n%2==0 else 112),map_y+(n//2)*57
            dx,dy = (x-x0)/scale,(y-y0)/scale
            transform = f'translate({dx},{dy})'
            axes[n].set('transform',transform)
            axes[n+4].set('transform',transform)
            # The same raw 0-to-unit-max scale remains encoded in each colorbar.
            text(layer,f'{condition}-unit-{int(row.unit)}',x+w/2,y-4,
                 [f'{int(row.unit)}:{row.color_max:.2g}'],30)
            label = find(layer,f'{condition}-unit-{int(row.unit)}')
            label.set('text-anchor','middle')
        if preserved_data(fields)!=prior:
            raise ValueError('Map content changed while relabeling')
        audit.append({'condition':condition,'figure':'fields','data_sha256':prior,
                      'selected_units':selected.unit.astype(int).tolist(),'font_pt':30})

        peak = find(root,fig_ids[1]); prior = preserved_data(peak)
        old_image = next(e for e in peak.iter() if e.tag == f'{{{SVG}}}image')
        x0,y0,w,h = bounds[old_image.get('id')]
        dx,dy = 190-x0,map_y-y0
        wrapper = ET.Element(f'{{{SVG}}}g',{'id':condition+'-peak-placement','transform':f'translate({dx},{dy})'})
        parent = next(p for p in root.iter() if peak in list(p)); place=list(parent).index(peak)
        parent.remove(peak);wrapper.append(peak);parent.insert(place,wrapper)
        strip_text(peak)
        text(layer,condition+'-peak-title',205,map_y-4,['DG peaks'],30)
        active = peaks[(peaks.condition==condition)&(peaks.seed==99)]
        labels = active[active.unit.isin(selected.unit)]
        for label in peak_label_positions(labels,190,map_y,w,h):
            ET.SubElement(layer,f'{{{SVG}}}line',{'id':f'{condition}-peak-label-leader-{label["unit"]}',
                'x1':str(label['point_x']),'y1':str(label['point_y']),
                'x2':str(label['x']+label['width']/2),'y2':str(label['y']-4),
                'stroke':'#505050','stroke-width':'.35'})
            text(layer,f'{condition}-peak-id-{label["unit"]}',label['x'],label['y'],[label['message']],30)
            el=find(layer,f'{condition}-peak-id-{label["unit"]}')
            el.set('style',el.get('style')+';paint-order:stroke;stroke:white;stroke-width:1.4px;stroke-linejoin:round')
        for value in (0,18):
            xx=190+(value+.5)/19*w
            yy=map_y+h-(value+.5)/19*h
            text(layer,f'{condition}-peak-x-{value}',xx-3,map_y+h+12,[str(value)],30)
            text(layer,f'{condition}-peak-y-{value}',174,yy+3,[str(value)],30)
        text(layer,condition+'-peak-x-unit',205,map_y+h+23,['x bin'],30)
        yy=map_y+h/2
        el=text(layer,condition+'-peak-y-unit',170,yy,['y bin'],30)
        el.set('transform',f'rotate(-90,170,{yy})')
        if preserved_data(peak)!=prior:
            raise ValueError('Peak data changed while relabeling')
        audit.append({'condition':condition,'figure':'peaks','data_sha256':prior,
                      'visible_peak_markers':len(active),'labeled_units':selected.unit.astype(int).tolist(),'font_pt':30})

        path = find(root,fig_ids[2]); prior = preserved_data(path)
        patch = next(e for e in path.iter() if e.get('id','').startswith('patch_2'))
        x0,y0,w,h = bounds[patch.get('id')]
        dx,dy=322-x0,map_y-y0
        wrapper=ET.Element(f'{{{SVG}}}g',{'id':condition+'-path-placement','transform':f'translate({dx},{dy})'})
        parent=next(p for p in root.iter() if path in list(p));place=list(parent).index(path)
        parent.remove(path);wrapper.append(path);parent.insert(place,wrapper)
        strip_text(path)
        for p in list(path.iter()):
            for c in list(p):
                if c.get('id','').startswith('legend_'):
                    p.remove(c)
        text(layer,condition+'-path-title',338,map_y-4,['Path'],30)
        for value,label in ((500,'0.5'),(2000,'2.0')):
            xx=322+(value-100)/1900*w
            yy=map_y+h-(value-100)/1900*h
            text(layer,f'{condition}-path-x-{value}',xx-5,map_y+h+12,[label],30)
            text(layer,f'{condition}-path-y-{value}',295,yy+3,[label],30)
        text(layer,condition+'-path-x-unit',320,map_y+h+26,['x (×10³ units)'],30)
        yy=map_y+h/2
        el=text(layer,condition+'-path-y-unit',288,yy,['y (×10³)'],30)
        el.set('text-anchor','middle')
        el.set('transform',f'rotate(-90,288,{yy})')
        if preserved_data(path)!=prior:
            raise ValueError('Trajectory vertices changed while relabeling')
        audit.append({'condition':condition,'figure':'trajectory','data_sha256':prior,
                      'all_vertices_retained':True,'font_pt':30})
    return audit


def import_existing(g: Gallery, name: str, source: Path, size):
    g.sources[str(source.relative_to(ROOT))]=sha(source)
    target=g.out/f'{name}.svg';target.write_bytes(source.read_bytes());g.figures.append(target)
    # Sources already use verified 30 pt typography at this physical size.
    root=ET.parse(target).getroot();root.set('width',f'{size[0]}mm');root.set('height',f'{size[1]}mm')
    ET.ElementTree(root).write(target,encoding='utf-8',xml_declaration=True)


def inserted_results(g: Gallery):
    data=g.read(TRANSFER/'reward_per_run.csv')
    fig,axes=scalar_axes()
    for ax,architecture in zip(axes,('D50','D51')):
        group=data[data.architecture==architecture]
        paired(ax,group,'full_0_75m',('W_WORKER','W_RAND_DG'),('Wkr','Rnd'),1000)
        for collection in ax.collections:collection.set_sizes([78])
        ax.set(title=f'{architecture} Reward\n/ step × 10³',ylim=(0,.6),yticks=[0,.3,.6])
        g.record('transfer_compact',architecture,'0–75M mean reward',group,'full_0_75m',1000,
                 protocol='joint-system transfer, three downstream seeds',dg_units=64)
    g.finish(fig,Candidate('transfer_compact','Joint-system transfer',(191.5,76.2),'Both versions','','','',''))
    path=ROOT/'06_experiments/results/A0_poster_analysis_20260926/c15_variants/trajectory_per_run.csv'
    raw=g.read(path)
    conditions=('CPD_C15_GATE_ACT_DIR','CPD_C15_GATE_ACT_DIR_GOAL')
    data=raw[raw.condition.isin(conditions)&raw.protocol.eq('frozen_10k_policy_probe')&
             raw.evaluation_age.eq(75_038_720)].copy()
    if len(data)!=6:raise ValueError('Expected three complete predictor seed pairs')
    fig,axes=scalar_axes()
    for ax,metric,title in zip(axes,('return_20_mobile','visited_fraction'),
                              ('Return (%)\n20 decisions','Visited\nbins (%)')):
        paired(ax,data,metric,conditions,('Off','On'),100)
        for collection in ax.collections:collection.set_sizes([78])
        ax.set(title=title,ylim=(0,100),yticks=[0,50,100])
        g.record('predictor',title,metric,data,metric,100,protocol='frozen_10k_policy_probe',
                 checkpoint_frames=75_038_720,dg_units=16)
    g.finish(fig,Candidate('predictor','Training-only DG predictor',(191.5,76.2),'Both versions','','','',''))
    data=g.read(TABLE)
    fig,axes=scalar_axes()
    for ax,metric,title in zip(axes,('target_hit_lift_terminal','action_sensitivity_terminal'),
                              ('Hit lift\n/ shuffled','Action-score\nchange')):
        dots(ax,data,metric,('C05','C15'))
        for collection in ax.collections:collection.set_sizes([78])
        ax.set_title(title)
        if metric.startswith('target'):
            ax.axhline(1,color='#777777',ls='--',lw=1.5)
            ax.set(ylim=(.8,1.2),yticks=[.8,1,1.2])
        else:ax.set(ylim=(0,.055),yticks=[0,.02,.04])
        g.record('goal_specificity',title,metric,data[data.condition.isin(('C05','C15'))],metric,
                 protocol='last_10m_training_scalars',checkpoint_frames=100_040_704,dg_units=16)
    g.finish(fig,Candidate('goal_specificity','Goal specificity',(191.5,76.2),'Both versions','','','',''))
    # Keep the concise C15 field-count claim tied to its actual frozen-map cohort.
    mono=g.read(FLAT/'mono_field_peak_counts.csv')
    selected=mono[mono.condition.isin(('C01','C05','C15'))]
    if len(selected)!=9 or selected.mono_units.sum()!=4 or not (selected.dg_units==16).all():
        raise ValueError('Strict-field comparison changed')
    method=FLAT/'mono_field_method.json';g.sources[str(method.relative_to(ROOT))]=sha(method)
    if json.loads(method.read_text())['min_dominant_component_mass_at_each_threshold']!=.8:
        raise ValueError('Strict field criterion changed')
    if selected.groupby('condition').mono_units.sum().to_dict()!={'C01':1,'C05':2,'C15':1}:
        raise ValueError('C01/C05/C15 single-field counts changed')
    selected.to_csv(g.out/'core_single_field_counts.csv',index=False)


def field_results(g: Gallery, survey_data: pd.DataFrame, *, selected_plots=None):
    """Compare both exploration and control without pooling incompatible measures."""
    binary=survey_data[survey_data.family.eq('CPD')].copy()
    if len(binary)!=81 or binary.condition_original.nunique()!=27 or not binary.frames.eq(75_005_952).all():
        raise ValueError('Expected the full 27-variant CPD family at its matched age')
    binary['condition']=binary.condition_original
    continuous=g.read(CONTINUOUS/'per_run.csv')
    continuous=continuous[continuous.family.eq('CPD')&
                          continuous.protocol.eq('online_latest_saved_window')].copy()
    if len(continuous)!=12 or not continuous.frames.eq(75_005_952).all():
        raise ValueError('Expected four locally available continuous-field CPD variants')
    statistics=[]
    for key,data,x,xlabel,factor in [
        ('monofields_behavior',binary,'mono_fraction_eligible','Single-field units (%)',100),
        ('dominance_behavior',continuous,'dominance','Field dominance',1),
    ]:
        if selected_plots is not None and key not in selected_plots:
            continue
        fig,axes=plt.subplots(1,2,figsize=tuple(v/25.4 for v in ROW_MM))
        fig.subplots_adjust(left=.16,right=.96,bottom=.43,top=.79,wspace=.50)
        for ax,y,title in zip(axes,('exploration_coverage','prospective_success'),('Exploration','Control')):
            ax.scatter(data[x]*factor,data[y]*100,color='#0072B2',s=78,alpha=.72,
                       edgecolor='white',linewidth=.5)
            ax.set(title=title,xlabel=xlabel,ylabel='Visited\n(%)' if y=='exploration_coverage' else 'DG hits\n(%)',
                   ylim=(0,100),yticks=[0,50,100])
            if factor==100:ax.set(xlim=(0,50),xticks=[0,25,50])
            else:ax.set(xlim=(0,1),xticks=[0,.5,1])
            ax.grid(color='#eeeeee')
            r=float(data[x].corr(data[y],method='spearman'))
            statistics.append({'plot':key,'metric':x,'outcome':y,'n':len(data),
                               'variants':data.condition.nunique(),'seeds_per_variant':3,
                               'checkpoint_frames':75_005_952,'spearman_rho':r})
            g.record(key,title,x,data,x,factor,dg_units=16,sample_unit='one training seed',x_axis=True)
            g.record(key,title,y,data,y,100,dg_units=16,sample_unit='one training seed',y_axis=True)
        g.finish(fig,Candidate(key,'Field structure versus behaviour',ROW_MM,'Section 3b','','','',''))
        data.to_csv(g.out/f'{key}_per_run.csv',index=False)
    (g.out/'field_behavior_statistics.json').write_text(json.dumps(statistics,indent=2)+'\n')


def predictor_role(g: Gallery, out: Path):
    """Pin the specific auxiliary head so the diagram cannot imply a planner."""
    exported=ROOT/'exports/intrmotiv_prediction_starter/intrmotiv_transfer/contextual_dg.py'
    provenance=exported.parents[1]/'PROVENANCE.json'
    study=ROOT/'hpc_runs/studies/ca3_feedback_predictive_dg.study.json'
    for path in (exported,provenance,study):g.sources[str(path.relative_to(ROOT))]=sha(path)
    code=exported.read_text()
    if 'self.predictor(torch.cat((source_dg, goal)' not in code or 'main_loss + control_loss' not in code:
        raise ValueError('Pinned transition predictor implementation changed; re-audit its role')
    role={'condition':'CPD_C15_GATE_ACT_DIR_GOAL',
          'setting':'dg_transition_prediction=goal', 'coefficient':.1,
          'input':['source DG activity','requested goal ID'],
          'target':'first distinct exclusive DG outcome, or timeout class',
          'training_only':True,'outputs_choose_actions_or_goals':False,
          'trained_by_main_loss':['DG projection','context-feedback adapter','predictor main head'],
          'trained_by_control_loss':['goal-only control head'],
          'history_gradient':'DIRECT detaches history; adapter weights remain trainable',
          'fixed_visual_trunk':True,
          'other_DG_learning':'existing ARR encoder and JOINT PPO gradients remain enabled',
          'pinned_module':str(exported.relative_to(ROOT)), 'pinned_module_sha256':sha(exported),
          'study_file':str(study.relative_to(ROOT)), 'study_file_sha256':sha(study),
          'verbatim_extraction_provenance':str(provenance.relative_to(ROOT))}
    runtime=Path('/home/xiaoxiong/SFgit/SF_hipposlam/sf_working_directories/IntrMotiv/dmlab/custom_learner.py')
    if runtime.exists():
        lines=runtime.read_text().splitlines()
        start=next(i for i,line in enumerate(lines) if 'transition_mode = getattr(self.cfg, "dg_transition_prediction"' in line)
        end=next(i for i in range(start,len(lines)) if 'encoder_loss *= self.cfg.encoder_grad_coeff' in lines[i])
        role['inspected_runtime_learner']={'path':str(runtime),'sha256':sha(runtime),
            'first_line':start+1,'last_line':end+1,'encoder_loss_excerpt':'\n'.join(lines[start:end+1])}
    (out/'predictor_role.json').write_text(json.dumps(role,indent=2)+'\n')


def scatter_panel(ax, data, x, title, x_label, *, marker_size=78):
    for family,group in data.groupby('family'):
        ax.scatter(group[x]*(100 if x=='graph_reachability' else 1),group.prospective_success*100,
                   c=FAMILY_COLORS[family],marker=FAMILY_MARKERS[family],s=marker_size,
                   alpha=.8,edgecolors='white',linewidth=.35,zorder=3)
    rho=float(data[x].corr(data.prospective_success,method='spearman'))
    ax.set(title=title,xlabel=x_label,ylim=(0,100),yticks=[0,50,100])
    if x=='spatial_information':ax.set(xlim=(0,.56),xticks=[0,.2,.4])
    elif x=='unique_peak_bins':ax.set(xlim=(6.5,16.8),xticks=[8,12,16])
    elif x=='graph_reachability':ax.set(xlim=(-2,102),xticks=[0,50,100])
    ax.grid(color='#e5e5e5',lw=.7,zorder=0)
    return rho


def survey(g: Gallery, size_mm=(383,130)):
    raw=g.read(SCATTER)
    data=raw[(raw.geometry_group=='legacy_19x19')&(raw.protocol=='online_latest_saved_window')&
             (raw.dg_units==16)].dropna(subset=['spatial_information','unique_peak_bins','prospective_success']).copy()
    if len(data)!=168 or data.condition.nunique()!=56 or data.run_name.duplicated().any():
        raise ValueError('The original fixed-DG-16 survey cohort changed')
    data['condition_original']=data.condition
    data['condition']=data.family
    statistics=[]
    fig,axes=plt.subplots(1,2,figsize=tuple(v/25.4 for v in size_mm),sharey=True)
    fig.subplots_adjust(left=.12,right=.975,top=.86 if size_mm[1]<=100 else .88,
                        bottom=.52 if size_mm[1]<=100 else .43,wspace=.28)
    for ax,x,label in zip(axes,('spatial_information','unique_peak_bins'),('Spatial score','Distinct peak bins')):
        rho=scatter_panel(ax,data,x,label,label)
        # Titles already name the horizontal measures. Removing the duplicate
        # labels leaves room for the six-family key at the original type size.
        ax.set_xlabel('')
        if size_mm[1]<=100:
            ax.set_title(f'{label} · ρ={rho:.2f}')
        else:
            ax.text(.04,.97,f'ρ={rho:.2f}',transform=ax.transAxes,va='top')
        statistics.append({'panel':'main_'+x,'sample_unit':'run','n':len(data),'rho':rho})
        g.record('survey_main',x,'recorded DG target-hit fraction',data,'prospective_success',100,
                 dg_units=16,protocol='historical counters at latest saved online spatial window')
    axes[0].set_ylabel('DG hits (%)')
    handles=[Line2D([],[],marker=FAMILY_MARKERS[f],color=FAMILY_COLORS[f],ls='none',markersize=8,
                    label=FAMILY_NAMES[f]) for f in sorted(data.family.unique())]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.015),ncol=3,frameon=False,
               handletextpad=.4,columnspacing=.9)
    g.finish(fig,Candidate('survey_main','Current section 3, compact layout',size_mm,'All versions','','','',''))
    (g.out/'cross_run_statistics.json').write_text(json.dumps(statistics,indent=2)+'\n')
    return data


def place_original(parent, group, x, y):
    copied=copy.deepcopy(group)
    s,_,_=matrix(copied)
    copied.set('transform',f'matrix({s},0,0,{s},{x},{y})')
    parent.append(copied)


def diagram_box(layer, ident, x, y, width, label, color='#253B57', height=18):
    """A schematic component, never a fabricated data point."""
    ET.SubElement(layer,f'{{{SVG}}}rect',{'id':ident+'-box','x':str(x),'y':str(y),
        'width':str(width),'height':str(height),'rx':'2','fill':'#F4F6F8',
        'stroke':color,'stroke-width':'.6'})
    el=text(layer,ident+'-label',x+width/2,y+12,[label],30,color=color)
    el.set('text-anchor','middle')


def diagram_arrow(layer, ident, x1, y1, x2, y2, color='#253B57', dashed=False):
    direction=np.array([x2-x1,y2-y1],dtype=float)
    direction/=np.linalg.norm(direction)
    normal=np.array([-direction[1],direction[0]])
    tip=np.array([x2,y2]);base=tip-direction*3
    attr={'id':ident,'x1':str(x1),'y1':str(y1),'x2':str(x2),'y2':str(y2),
          'stroke':color,'stroke-width':'.8'}
    if dashed:attr['stroke-dasharray']='2,1.5'
    ET.SubElement(layer,f'{{{SVG}}}line',attr)
    left,right=base+normal*1.6,base-normal*1.6
    ET.SubElement(layer,f'{{{SVG}}}path',{'id':ident+'-head',
        'd':f'M {tip[0]},{tip[1]} L {left[0]},{left[1]} L {right[0]},{right[1]} Z','fill':color})


def control_diagram(layer, x, y):
    text(layer,'control-schematic-title',x,y+10,['Same state; change goal'],30,bold=True)
    for index,goal in enumerate(('A','B')):
        yy=y+19+index*25
        diagram_box(layer,'control-goal-'+goal,x,yy,40,'Goal '+goal)
        diagram_arrow(layer,'control-input-'+goal,x+42,yy+9,x+49,yy+9)
        diagram_box(layer,'control-policy-'+goal,x+50,yy,52,'Policy')
        diagram_arrow(layer,'control-output-'+goal,x+104,yy+9,x+111,yy+9)
        diagram_box(layer,'control-scores-'+goal,x+112,yy,67,'Scores')
    text(layer,'control-schematic-metric',x,y+75,['Compare mean |Δlogit|'],30)


def transfer_diagram(layer, x, y):
    for index,(name,labels,color) in enumerate([
        ('WORKER',('DG','Worker','Graph'),'#0072B2'),
        ('RAND_DG',('Random','Fresh','Empty'),'#D55E00')]):
        yy=y+index*38
        text(layer,'transfer-'+name+'-title',x,yy+9,[name],30,bold=True,color=color)
        for i,(offset,width,label) in enumerate(zip((0,60,122),(50,52,56),labels)):
            diagram_box(layer,f'transfer-{name}-{i}',x+offset,yy+15,width,label,color)
            if i<2:diagram_arrow(layer,f'transfer-{name}-arrow-{i}',x+offset+width+2,yy+24,
                                  x+(60,122)[i]-2,yy+24,color)


def predictor_diagram(layer, x, y):
    text(layer,'predictor-schematic-title',x,y+10,['Training only'],40,bold=True)
    diagram_box(layer,'predictor-input',x+20,y+18,159,'DG state + goal')
    diagram_arrow(layer,'predictor-forward',x+100,y+38,x+100,y+44)
    diagram_box(layer,'predictor-output',x+20,y+46,159,'Next event / timeout')
    # The return arrow marks learning, not an online planning loop.
    diagram_arrow(layer,'predictor-gradient',x+9,y+57,x+9,y+28,color='#D55E00',dashed=True)
    text(layer,'predictor-gradient-label',x,y+75,['Loss trains DG + context'],30,color='#D55E00')


def compose_right(root, source_right, g: Gallery, variant_key):
    item=VARIANTS[variant_key];x=442.4
    remove(root,find(root,'poster-right-20260929'))
    layer=ET.SubElement(root,f'{{{SVG}}}g',{'id':'revised-right-'+variant_key,
        f'{{{INK}}}groupmode':'layer',f'{{{INK}}}label':'Results · '+item['title']})
    section(layer,2,'Explore, transfer and predict',x,220)
    text(layer,'subsection-2a',x,244,['2a  Exploration and revisits'],36,bold=True,color='#253B57')
    text(layer,'design-key',x,263,['C01 no goals · C05 uniform goals · C15 frontier goals'],30)
    place_original(layer,find(source_right,'right-exploration-plot'),x,272)
    text(layer,'coverage-gain',x,363,['C15: 2.1× coverage; no more single fields.'],40,bold=True)
    text(layer,'subsection-2b',x,389,['2b  Does the requested goal change behaviour?'],36,bold=True,color='#253B57')
    embed(layer,g.out/'goal_specificity.svg','goal-specificity',x,400,191.5)
    control_diagram(layer,645.4,400)
    text(layer,'goal-control-caption',x,490,['C15: little goal-specific advantage.'],40)
    text(layer,'control-reference',x,509,['100M · 3 seeds · hit lift = issued / shuffled · 1 = no advantage'],30)
    text(layer,'subsection-2c',x,534,['2c  Transfer the learned system'],36,bold=True,color='#253B57')
    embed(layer,g.out/'transfer_compact.svg','transfer-result',x,545,191.5)
    transfer_diagram(layer,645.4,545)
    text(layer,'transfer-caption',x,637,['Mean reward +6% / +16%; 2/3 seed pairs win.'],40)
    text(layer,'transfer-protocol',x,655,['75M task frames · DG 64 frozen · pretrained worker adapts'],30)
    text(layer,'subsection-2d',x,680,['2d  Train DG to predict the next landmark'],36,bold=True,color='#253B57')
    embed(layer,g.out/'predictor.svg','predictor-result',x,691,191.5)
    predictor_diagram(layer,645.4,691)
    text(layer,'predictor-caption',x,783,['More revisits, less coverage in all 3 seed pairs.'],40)
    text(layer,'predictor-protocol',x,800,['75M · DG 16 · 10k probe · Off/On: auxiliary predictor loss'],30)

    section(layer,3,'Field shape and behaviour',x,831)
    text(layer,'survey-protocol',x,858,['3a  DG 16 · 168 runs / 56 variants · 25–150M'],36,bold=True,color='#253B57')
    embed(layer,g.out/'survey_main.svg','survey-main',x,870,383)
    text(layer,'subsection-3b',x,1023,['3b  Single fields and behavioural performance'],36,bold=True,color='#253B57')
    embed(layer,g.out/f"{item['field_plot']}.svg",'field-behavior',x,1034,383)
    text(layer,'field-message',x,1127,[item['field_caption']],40,bold=True)
    text(layer,'field-protocol',x,1147,[item['field_protocol']],30)
    text(layer,'field-definition',x,1162,
         ['Eligible units · DG hits are internal · associations only'],30)
    text(layer,'references',x,1178,['Lin, Yiu & Leibold (2026) · Leibold (2020), Neural Networks.'],30)


def export_svg(root, path):
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)


def render(path, width=1800):
    result=subprocess.run(['inkscape',str(path),'--export-area-page',f'--export-width={width}',
                           f'--export-filename={path.with_suffix(".png")}'],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)


def check_svg(path):
    root=ET.parse(path).getroot()
    ids=[e.get('id') for e in root.iter() if e.get('id')]
    if len(ids)!=len(set(ids)):raise ValueError(f'Duplicate SVG IDs: {path}')
    refs={ref for e in root.iter() for value in e.attrib.values() for ref in re.findall(r'url\(#([^)]+)\)',value)}
    if refs.difference(ids):raise ValueError('Unresolved SVG references')
    return {'file':path.name,'unique_ids':True,'resolved_references':True}


def overview(out: Path):
    """A preview sheet; full-size editable A0 SVGs remain the deliverables."""
    width=20+840*len(VARIANTS)
    root=ET.Element(f'{{{SVG}}}svg',{'width':str(width),'height':'1320','viewBox':f'0 0 {width} 1320'})
    ET.SubElement(root,f'{{{SVG}}}rect',{'width':str(width),'height':'1320','fill':'white'})
    title=ET.SubElement(root,f'{{{SVG}}}text',{'x':'20','y':'50','style':'font-family:DejaVu Sans;font-size:38px;font-weight:bold'})
    title.text='Two poster alternatives · transfer and predictor in both'
    for i,(key,item) in enumerate(VARIANTS.items()):
        x=20+i*840
        label=ET.SubElement(root,f'{{{SVG}}}text',{'x':str(x),'y':'105','style':'font-family:DejaVu Sans;font-size:32px;font-weight:bold'})
        label.text=key[0]+' · '+item['title']+(' (recommended)' if item['recommended'] else '')
        png=out/f'Bernstein2026_{key}.png'
        ET.SubElement(root,f'{{{SVG}}}image',{'x':str(x),'y':'125','width':'800','height':str(800*1189/841),
            'href':'data:image/png;base64,'+base64.b64encode(png.read_bytes()).decode()})
    target=out/'comparison_overview.svg';export_svg(root,target);render(target,width)


def visual_checks(out: Path, source: Path):
    """Verify protected pixels and actual on-page type bounds after Inkscape."""
    from PIL import Image
    reference=Path('/tmp')/('intrmotiv-layout-source-'+sha(source)[:12]+'.png')
    if not reference.exists():
        subprocess.run(['inkscape',str(source),'--export-area-page','--export-width=1800',
                        f'--export-filename={reference}'],capture_output=True,check=True)
    original=np.asarray(Image.open(reference).convert('RGBA'))
    previous=np.asarray(Image.open(PREVIOUS/'Bernstein2026_A_system_transfer.png').convert('RGBA'))
    results=[]
    for key in VARIANTS:
        path=out/f'Bernstein2026_{key}.svg'
        actual=np.asarray(Image.open(path.with_suffix('.png')).convert('RGBA'))
        regions={}
        for name,box in [('header',(0,0,841,193)),('introduction',(0,193,431,720))]:
            x0,y0,x1,y1=[round(v/841*1800) for v in box]
            changed=int(np.any(original[y0:y1,x0:x1]!=actual[y0:y1,x0:x1],axis=2).sum())
            if changed:raise ValueError(f'Protected {name} pixels changed in {key}: {changed}')
            regions[name]=changed
        left_edge=round(431/841*1800)
        left_changes=int(np.any(previous[:,:left_edge]!=actual[:,:left_edge],axis=2).sum())
        if left_changes:raise ValueError(f'Previously delivered left column changed in {key}: {left_changes}')
        root=ET.parse(path).getroot();bounds=query_bounds(path)
        added=[find(root,'revised-left-typography'),find(root,'revised-right-'+key)]
        clipped=[]
        for layer in added:
            for el in layer.iter(f'{{{SVG}}}text'):
                box=bounds.get(el.get('id'))
                if box is None:continue
                x,y,w,h=box
                if x<0 or y<0 or x+w>841 or y+h>1189:
                    clipped.append({'id':el.get('id'),'box_mm':box.tolist()})
        if clipped:raise ValueError(f'Added text extends beyond the A0 page: {clipped}')
        # Selected peak labels and column titles must not collide.
        collisions=[]
        for condition in LEFT_FIGURES:
            elements=[e for e in added[0].iter(f'{{{SVG}}}text')
                      if e.get('id','').startswith(condition+'-peak-id-') or e.get('id')==condition+'-peak-title']
            for i,left in enumerate(elements):
                for right in elements[i+1:]:
                    a,b=bounds[left.get('id')],bounds[right.get('id')]
                    overlap=np.minimum(a[:2]+a[2:],b[:2]+b[2:])-np.maximum(a[:2],b[:2])
                    if (overlap>.2).all():collisions.append([left.get('id'),right.get('id')])
        if collisions:raise ValueError(f'Peak labels overlap: {collisions}')
        # Check actual glyph boxes across all newly composed right-column text,
        # including imported plot labels and captions next to diagrams.
        right_text=[e for e in added[1].iter(f'{{{SVG}}}text') if e.get('id') in bounds]
        right_collisions=[]
        for i,left in enumerate(right_text):
            for right in right_text[i+1:]:
                a,b=bounds[left.get('id')],bounds[right.get('id')]
                overlap=np.minimum(a[:2]+a[2:],b[:2]+b[2:])-np.maximum(a[:2],b[:2])
                if (overlap>.3).all():right_collisions.append([left.get('id'),right.get('id')])
        if right_collisions:raise ValueError(f'Right-column text overlaps: {right_collisions}')
        results.append({'variant':key,'protected_pixel_changes':regions,
                        'previous_left_column_pixel_changes':left_changes,'added_text_on_page':True,
                        'peak_labels_do_not_overlap':True,'right_text_do_not_overlap':True,'svg_sha256':sha(path)})
    (out/'quality_checks.json').write_text(json.dumps(results,indent=2)+'\n')


def write_notes(out: Path):
    lines=['# Two revised A0 posters: transfer, predictor, and field structure','',
        'Both versions combine the transfer result from the former version 1 with the predictor '
        'result from version 2. The former version 3 is dropped from the current selection. '
        'The header and entire left column match the previous recommended poster.','',
        '![Two-version overview](comparison_overview.png)','',
        '| Version | Section 3b | Recommendation |',
        '| --- | --- | --- |',
        '| [A: single-field fraction](Bernstein2026_A_transfer_predictor_monofields.svg) | Exploration and control across all 81 CA3-feedback runs | Preferred: broadest direct check of the monofield message |',
        '| [B: continuous field dominance](Bernstein2026_B_transfer_predictor_dominance.svg) | Same two outcomes across 12 locally available runs | Alternative: avoids the binary field cutoff |','',
        'SVGs are editable A0, 841 × 1189 mm. Plot text and diagram labels are 30 pt; '
        'conclusion captions are 40 pt. New markers are 78 pt². The overview is a reduced preview. '
        'Existing 383 × 76.2 mm and 191.5 × 76.2 mm canvases are retained.','',
        '## Common content','',
        '**2a Exploration and revisits.** C15 has 2.1× the final-window coverage AUC of C01. '
        'Frozen maps at that checkpoint have one strict single-field unit out of 48 across three seeds '
        'in both C01 and C15; C05 has two. This supports improvement without increased monofield counts, '
        'not a causal claim that monofields have no benefit. Returns count mobile path windows with '
        'path >500 units and displacement <100; the plots distinguish 20 and 40 decisions.','',
        '**2b Goal specificity.** The two small plots retain issued-versus-shuffled target-hit lift '
        'and mean absolute raw-logit change under a goal swap. The diagram holds state fixed and '
        'changes the requested goal, making action sensitivity understandable. Lift 1 is the '
        'shuffled reference. Action scores are raw logits, not probabilities. The diagram is '
        'schematic and contains no fabricated measurements. The caption now makes one short '
        'point: C15 has little goal-specific advantage in this diagnostic.','',
        '**2c Transfer.** WORKER transfers learned DG, worker and graph; RAND_DG starts with random '
        'DG, a fresh worker and an empty graph. Both use frozen DG 64 and 75M downstream frames. '
        'The package diagram replaces the long explanatory paragraph. Full-run mean reward '
        'gains are +6.1% for D50 and +15.8% for D51; WORKER wins two of three downstream seed '
        'pairs in each. Points are training seeds, lines pair the seeds and black bars are means. '
        'This tests the joint system and includes source pretraining. It does not isolate DG '
        'learning, show an immediate head start, or supply a matched WORKER heldout evaluation. '
        'See [the transfer evidence](../../worker_random_transfer/README.md).','',
        '**2d Training-only goal-conditioned outcome prediction.** This is the CPD '
        '`dg_transition_prediction=goal` head, not the historical shadow CA3 predictor or the later '
        'predictive-state readout. It receives source DG activity and the requested goal and predicts '
        'the first distinct exclusive DG outcome or timeout. Cross-entropy is added to the encoder '
        'loss with coefficient 0.1. The main branch trains the DG projection, context-feedback '
        'adapter and predictor head; the goal-only control trains its own head. DIRECT mode '
        'detaches the input history. The fixed ImageNet trunk stays fixed, and predictor outputs '
        'do not select actions or manager goals. DG also receives its existing ARR and joint PPO '
        'training gradients. The return arrow in the diagram means a training gradient, not planning.','',
        'The Off/On contrast uses the same GATE_ACT_DIR design, with and without that auxiliary '
        'loss. Frozen 10k probes at 75,038,720 frames show more 20-decision mobile returns '
        '(62.4% → 83.2%) and less visited-bin coverage (69.3% → 55.3%), in all three seed pairs. '
        'Starts and stochastic paths are not identical. Reinforcing familiar cycles is a '
        'hypothesis; the result is not a general verdict on prediction. '
        '[predictor_role.json](predictor_role.json) pins the role, study and inspected source evidence.','',
        '**3a Architecture context.** The original two survey measures and all 168 runs / 56 '
        'variants remain. DG capacity stays at 16. Families differ in ages and outcome rules; '
        'spatial score and peak diversity should not be treated as causal control evidence.','',
        '## The monofield message covers exploration and control','',
        '> **More single-field structure does not reliably track better behaviour in these comparisons.**','',
        'Version A uses the complete CA3-feedback family: 27 variants × three seeds, all '
        '75,005,952 frames. Single-field fraction is the fraction of eligible units passing '
        'the existing 80% dominant-component criterion at every 30/50/70% peak threshold. '
        'Its rank association is $\rho=-0.064$ with visited coverage and $\rho=-0.324$ with '
        'recorded target hits. The x axis displays 0–50% (all observed values fit); y axes '
        'retain 0–100%. Coverage spans only 84.2–88.1%, so there is little headroom in this '
        'particular exploration measure.','',
        'Version B retains a continuous score before classification: minimum dominant-component '
        'mass over those thresholds, averaged over eligible units. The four locally available '
        'CA3-feedback variants give 12 seed-level points: $\rho=-0.616$ for online coverage and '
        '$\rho=-0.266$ for target hits. The 0–1 x axis and 0–100% y axes are retained. '
        'This score is not spatial concentration and not mean-threshold dominance. '
        '[Continuous-field evidence](../../continuous_fields/README.md) records the broader '
        'availability and protocol sensitivity.','',
        'Both views show architectural associations, not interventions on field shape. '
        'Recent policy-driven maps and historical internal target-event counters cover different '
        'time spans; DG target hits do not verify physical arrival. A weak association cannot '
        'establish equivalence or prove that single fields are unnecessary. The shorter public '
        'caption expresses the observed dissociation without claiming that monofields cause harm.','',
        '## Verification and reuse','',
        '[manifest.json](manifest.json) fingerprints sources and all nine retained left data groups. '
        '[quality_checks.json](quality_checks.json) checks protected pixels, page bounds and '
        'text collisions. [Plotted points](plots/plotted_points.csv) and '
        '[field statistics](plots/field_behavior_statistics.json) retain exact coordinates, '
        'cohorts and measurement definitions. No training, cluster access or new telemetry was used.','',
        '```bash',
        '/home/xiaoxiong/miniforge3/envs/SF_git/bin/python 06_experiments/revise_poster_layout_20260929.py --source 05_plans/poster_20260929/layout_versions/input_layout.svg',
        '```','',
        '**Reusable experience:** reuse the pinned survey and saved trajectories, verify '
        'the exact predictor variant before drawing gradient arrows, and regenerate compact '
        'plots at their intended physical size. Compare the entire left column against the '
        'previous delivered poster, not just the older attachment. Keep diagram labels short '
        'and inspect actual Inkscape glyph bounds after rendering.','']
    (out/'README.md').write_text('\n'.join(lines))


def helvetica_text(group):
    """Change editable figure labels, retaining sizes, weights and data paths."""
    for element in group.iter(f'{{{SVG}}}text'):
        for node in element.iter():
            css=node.get('style','')
            css=re.sub(r'font-family\s*:[^;]+;?', '', css)
            css=re.sub(r'-inkscape-font-specification\s*:[^;]+;?', '', css)
            node.set('style',css.rstrip(';')+';font-family:Helvetica,Nimbus Sans')
            if 'font-family' in node.attrib:
                node.set('font-family','Helvetica,Nimbus Sans')


def command_diagnostics(g: Gallery):
    """Reuse exact saved commands and disclose the smaller TV overlap.

    CPD has no exported TV diagnostic. TV here is a training-window goal-swap
    statistic in the credit-assignment family. Command lift uses frozen
    intervention trials; neither is interchangeable with historical DG hits.
    """
    commands=g.read(CONTINUOUS/'command_dominance_per_run.csv')
    if len(commands)!=12 or commands.run_name.duplicated().any() or not commands.frames.eq(75_038_720).all():
        raise ValueError('Expected twelve exact common-replay/command models')
    if commands.panel.nunique()!=1 or not commands.dg_units.eq(16).all():
        raise ValueError('Command cohort capacity or shared replay panel changed')
    if not np.allclose(commands.command_lift,
                       commands.executed_success_command-commands.matched_shuffled_success_command):
        raise ValueError('Command lift must be an absolute success difference')
    terminal=ROOT/'06_experiments/results/A0_poster_analysis_20260926/c15_variants/source_inputs/06_experiments/results/recent_batches_audit_20260906/saturday_terminal_per_run.csv'
    online=g.read(terminal);online['run_name']=online.run_name.str.strip()
    joined=commands.merge(online[['run_name','action_probability_tv','window_start','max_env_steps']],
                          on='run_name',how='left',validate='one_to_one')
    tv=joined.dropna(subset=['action_probability_tv']).copy()
    if (len(tv)!=6 or not tv.family.eq('Saturday').all() or not tv.window_start.eq(70_000_000).all()
            or not tv.max_env_steps.eq(74_973_180).all()):
        raise ValueError('Expected six overlapping credit models with saved TV')
    commands.to_csv(g.out/'command_diagnostics_per_run.csv',index=False)
    tv.to_csv(g.out/'action_tv_per_run.csv',index=False)

    groups=(('DGP',('DGP_C15_FIRST_JOINT_LEG','DGP_C15_HIT_JOINT_LEG'),('First','Hit')),
            ('Saturday',('SAT_C15_SRC_MON_FILM','SAT_C15_ARR_MON_FILM'),('Source','Arrival')))
    fig,axes=scalar_axes()
    for ax,(family,conditions,labels) in zip(axes,groups):
        data=commands[commands.family.eq(family)]
        paired(ax,data,'command_lift',conditions,labels,100)
        for collection in ax.collections:collection.set_sizes([78])
        ax.set(title='DG policy' if family=='DGP' else 'Credit',
               ylim=(0,35),yticks=[0,15,30])
        ax.axhline(0,color='#777777',ls='--',lw=1)
        g.record('command_lift',family,'executed minus matched shuffled success',data,'command_lift',100,
                 dg_units=16,checkpoint_frames=75_038_720,protocol='frozen_matched_commands',units='percentage points')
    fig.supylabel('Lift (p.p.)',fontsize=30)
    g.finish(fig,Candidate('command_lift','Commanded success',(191.5,76.2),'Evolved B','','','',''))

    fig,ax=plt.subplots(figsize=(191.5/25.4,76.2/25.4),layout='constrained')
    paired(ax,tv,'action_probability_tv',groups[1][1],groups[1][2],100)
    for collection in ax.collections:collection.set_sizes([78])
    ax.set(title='Action TV · 6 credit models',ylabel='TV (%)',ylim=(0,.5),yticks=[0,.25,.5])
    g.record('action_tv','Credit','action_probability_tv',tv,'action_probability_tv',100,
             dg_units=16,protocol='last_5m_training_scalars',training_window_start=70_000_000,
             units='percent probability mass')
    g.finish(fig,Candidate('action_tv','Goal-swap probability sensitivity',(191.5,76.2),'Evolved B','','','',''))

    statistics=[]
    for key,outcome,datasets in [
        ('dominance_command','command_lift',[commands[commands.family.eq(f)] for f,_,_ in groups]),
        ('dominance_tv','action_probability_tv',[tv]),
    ]:
        size_mm=(383 if len(datasets)==2 else 191.5,76.2)
        fig,axes=plt.subplots(1,len(datasets),figsize=tuple(v/25.4 for v in size_mm),layout='constrained',squeeze=False)
        for ax,data in zip(axes.flat,datasets):
            family=data.family.iloc[0]
            _,conditions,labels=next(row for row in groups if row[0]==family)
            for index,(condition,label) in enumerate(zip(conditions,labels)):
                rows=data[data.condition.eq(condition)]
                ax.scatter(rows.mean_threshold_dominance,rows[outcome]*100,s=78,
                    color=('#0072B2','#D55E00')[index],marker=('o','s')[index],label=label)
            ax.set(title='DG policy' if family=='DGP' else 'Credit',
                   xlabel='Mean field dominance',ylabel='Lift (p.p.)' if outcome=='command_lift' else 'TV (%)',
                   xlim=(0,1),xticks=[0,.5,1],ylim=(0,35) if outcome=='command_lift' else (0,.5))
            ax.set_yticks([0,15,30] if outcome=='command_lift' else [0,.25,.5])
            ax.grid(color='#eeeeee')
            # Scores occupy x > 0.42. A short vertical key at the left leaves
            # all six observations visible; a horizontal key covered points.
            ax.legend(frameon=False,ncol=1,loc='upper left',handlelength=.7,
                      handletextpad=.35,borderpad=.2,labelspacing=.2)
            rho=float(data.mean_threshold_dominance.corr(data[outcome],method='spearman'))
            statistics.append({'plot':key,'family':family,'n':len(data),'variants':2,
                               'spearman_rho':rho,'definition':'mean-threshold dominance on common replay'})
            g.record(key,family,'mean_threshold_dominance',data,'mean_threshold_dominance',
                     x_axis=True,protocol='common_replay',dg_units=16,checkpoint_frames=75_038_720)
            g.record(key,family,outcome,data,outcome,100,y_axis=True,units='p.p.' if outcome=='command_lift' else '%')
        g.finish(fig,Candidate(key,'Field dominance and controllability',size_mm,'Supplementary','','','',''))
    stats={'command_models':12,'tv_models':6,'all_command_lifts_positive':bool(commands.command_lift.gt(0).all()),
        'command_lift_pp_range':[float(commands.command_lift.min()*100),float(commands.command_lift.max()*100)],
        'action_tv_percent_range':[float(tv.action_probability_tv.min()*100),float(tv.action_probability_tv.max()*100)],
        'field_associations':statistics,
        'action_tv_definition':'0.5 * sum(abs(softmax(original_logits) - softmax(alternate_logits))); '
                               'alternate goal is rolled over the same minibatch; mean over valid original goals',
        'tv_window':'70M to 74,973,180 training frames; not the frozen command probe',
        'command_definition':'executed internal DG success minus retrospectively matched shuffled-command success',
        'limitations':['CPD TV is unavailable; no imputation','CPD minimum-threshold dominance and replay mean-threshold dominance stay separate',
                       'Shuffled trials do not guarantee identical physical starts; ordered-pair coverage is incomplete',
                       'DG hits do not verify physical arrival; these are architecture/seed associations']}
    runtime=Path('/home/xiaoxiong/SFgit/SF_hipposlam/sf_working_directories/IntrMotiv/dmlab/custom_learner.py')
    if runtime.exists():
        lines=runtime.read_text().splitlines()
        start=next(i for i,line in enumerate(lines) if line.startswith('def categorical_action_total_variation('))
        stats['tv_source_audit']={'path':str(runtime),'sha256':sha(runtime),'first_line':start+1,
                                'excerpt':'\n'.join(lines[start:start+8])}
    (g.out/'command_diagnostics_statistics.json').write_text(json.dumps(stats,indent=2)+'\n')
    return stats


def evolved_right(root, source_right, g: Gallery):
    """Retain B's message, make room for separately labelled command evidence."""
    key='B_transfer_predictor_dominance'
    compose_right(root,source_right,g,key)
    layer=find(root,'revised-right-'+key)
    positions={'subsection-2b':386,'goal-specificity':397,'goal-control-caption':486,
        'subsection-2c':514,'transfer-result':525,'transfer-caption':614,
        'subsection-2d':642,'predictor-result':653,'predictor-caption':742,
        'section-3-badge':753,'section-3-number':770,'section-3-title':770,
        'survey-protocol':794,'survey-main':803,'subsection-3b':922,'field-behavior':933,
        'field-message':1025,'field-protocol':1043}
    for ident,y in positions.items():
        node=find(layer,ident)
        if ident.endswith(('result','specificity','main','behavior')):
            transform=node.get('transform','')
            old_y=float(re.search(r'translate\([^,]+,\s*([\d.]+)',transform)[1])
        else:old_y=float(node.get('y'))
        # A wrapper shifts all descendants, including tspan baselines.
        parent=next(p for p in layer.iter() if node in list(p));index=list(parent).index(node)
        parent.remove(node)
        wrapper=ET.Element(f'{{{SVG}}}g',{'id':ident+'-shift','transform':f'translate(0,{y-old_y})'})
        wrapper.append(node);parent.insert(index,wrapper)
    for prefix,dy in [('control-',-3),('transfer-WORKER',-20),('transfer-RAND_DG',-20),('predictor-',-38)]:
        nodes=[node for node in list(layer) if node.get('id','').startswith(prefix)]
        for node in nodes:
            # Captions already moved through their own wrapper.
            if node.get('id','').endswith('-shift'):continue
            parent=layer;index=list(parent).index(node);parent.remove(node)
            wrapper=ET.Element(f'{{{SVG}}}g',{'id':node.get('id')+'-shift','transform':f'translate(0,{dy})'})
            wrapper.append(node);parent.insert(index,wrapper)
    for ident in ('control-reference','transfer-protocol','predictor-protocol','field-definition','references'):
        remove(layer,find(layer,ident))
    for line in find(layer,'field-protocol').iter(f'{{{SVG}}}tspan'):
        line.text='CA3 feedback · 4 × 3 seeds · DG 16 · 75M · associations only'
    text(layer,'subsection-3c',442.4,1066,['3c  Goal sensitivity and commanded success'],36,bold=True,color='#253B57')
    embed(layer,g.out/'command_lift.svg','command-lift',442.4,1077,191.5)
    embed(layer,g.out/'action_tv.svg','action-tv',633.9,1077,191.5)
    text(layer,'command-message',442.4,1168,['TV <0.5%; command lift +11–29 p.p. · 3 seeds'],40,bold=True)
    ET.SubElement(layer,f'{{{SVG}}}rect',{'id':'references-box','x':'441.4','y':'1174',
        'width':'385','height':'14','rx':'1.5','fill':'#F4F6F8','stroke':'#253B57','stroke-width':'.6'})
    text(layer,'references',445.4,1185,['References: Lin, Yiu & Leibold (2026); Leibold (2020), Neural Networks.'],30)
    return layer


def trajectory_legend(root):
    """Shared key above all three trajectory panels; actual events do not move."""
    layer=find(root,'revised-left-typography')
    label=find(layer,'left-common-protocol')
    for child in list(label):label.remove(child)
    ET.SubElement(label,f'{{{SVG}}}tspan',{'x':'29','y':'759'}).text='Seed 99 · map labels: unit : max'
    ET.SubElement(layer,f'{{{SVG}}}circle',{'id':'trajectory-option-start-key','cx':'292','cy':'755.5',
        'r':'2.5','fill':'none','stroke':'#0B69A3','stroke-width':'.7'})
    text(layer,'trajectory-option-start-label',298,759,['Option start'],30)
    # Same star encoding as the saved DG event markers.
    angles=np.arange(10)*np.pi/5-np.pi/2
    radii=np.array([3.5,1.45]*5)
    vertices=np.column_stack((365+np.cos(angles)*radii,755.5+np.sin(angles)*radii))
    ET.SubElement(layer,f'{{{SVG}}}polygon',{'id':'trajectory-goal-hit-key',
        'points':' '.join(f'{x},{y}' for x,y in vertices),'fill':'#D38A00','stroke':'#754B00','stroke-width':'.35'})
    text(layer,'trajectory-goal-hit-label',371,759,['Goal hit'],30)


def evolve_b(out: Path):
    """A scoped evolution of the selected B, preserving all existing data."""
    base=DEFAULT_OUT/'Bernstein2026_B_transfer_predictor_dominance.svg'
    digest=sha(base);out.mkdir(parents=True,exist_ok=True)
    font=style('Helvetica',fallback_family='Nimbus Sans')
    plots=out/'plots';plots.mkdir(exist_ok=True);g=Gallery(plots)
    for path in (base,Path(__file__),Path(__file__).with_name('compose_a0_poster_20260929.py'),
                 Path(__file__).with_name('render_poster_candidate_plots_20260929.py')):
        g.sources[str(path.relative_to(ROOT))]=sha(path)
    data=survey(g,size_mm=(383,100));inserted_results(g)
    field_results(g,data,selected_plots={'dominance_behavior'});predictor_role(g,out)
    diagnostics=command_diagnostics(g)
    for path in g.figures:
        figure=ET.parse(path).getroot();helvetica_text(figure);export_svg(figure,path)
    # compose_right needs only the retained original exploration panel. The
    # selected B already has the revised left column, so do not rebuild it.
    root=ET.parse(base).getroot()
    source_right=ET.Element(f'{{{SVG}}}g',{'id':'right-source'})
    source_right.append(copy.deepcopy(find(root,'right-exploration-plot')))
    previous_right=find(root,'revised-right-B_transfer_predictor_dominance')
    previous_right.set('id','poster-right-20260929')
    layer=evolved_right(root,source_right,g)
    trajectory_legend(root)
    helvetica_text(find(root,'revised-left-typography'));helvetica_text(layer)
    target=out/'Bernstein2026_B_evolved.svg';export_svg(root,target);render(target)
    verify_and_render(g,base,digest)
    pd.DataFrame(g.points).to_csv(plots/'plotted_points.csv',index=False)
    from PIL import Image
    before=np.asarray(Image.open(base.with_suffix('.png')).convert('RGBA'))
    after=np.asarray(Image.open(target.with_suffix('.png')).convert('RGBA'))
    protected={}
    for name,box in [('header',(0,0,841,193)),('introduction',(0,193,431,720))]:
        x0,y0,x1,y1=[round(v/841*1800) for v in box]
        changed=int(np.any(before[y0:y1,x0:x1]!=after[y0:y1,x0:x1],axis=2).sum())
        if changed:raise ValueError(f'Protected {name} changed: {changed}')
        protected[name]=changed
    original=ET.parse(base).getroot();audit=[]
    for condition,ids in LEFT_FIGURES.items():
        for ident in ids:
            prior=preserved_data(find(original,ident));current=preserved_data(find(root,ident))
            if prior!=current:raise ValueError(f'Left plotted data changed: {ident}')
            audit.append({'condition':condition,'figure':ident,'data_sha256':current})
    bounds=query_bounds(target)
    added=[find(root,'revised-left-typography'),layer]
    clipped=[];collisions=[]
    for group in added:
        labels=[el for el in group.iter(f'{{{SVG}}}text') if el.get('id') in bounds]
        for i,label in enumerate(labels):
            a=bounds[label.get('id')]
            if a[0]<0 or a[1]<0 or a[0]+a[2]>841 or a[1]+a[3]>1189:clipped.append(label.get('id'))
            for other in labels[i+1:]:
                b=bounds[other.get('id')]
                overlap=np.minimum(a[:2]+a[2:],b[:2]+b[2:])-np.maximum(a[:2],b[:2])
                if (overlap>.3).all():collisions.append([label.get('id'),other.get('id')])
    if clipped or collisions:raise ValueError(f'Text quality: clipped={clipped}, overlaps={collisions}')
    for group in added:
        if any('font-family:Helvetica,Nimbus Sans' not in e.get('style','') for e in group.iter(f'{{{SVG}}}text')):
            raise ValueError('Generated figure label lacks Helvetica declaration')
    report={'schema':'intrmotiv/poster-layout-evolution/v1','workflow_version':WORKFLOW_VERSION,
        'study_schema':'intrmotiv/study/v1','selected_base':str(base.relative_to(ROOT)),'base_sha256':digest,
        'font_requested':'Helvetica','font_rendered':'Nimbus Sans','font_path':font,
        'font_substitution':'Helvetica absent locally; verified scalable Nimbus Sans used explicitly',
        'plot_font_pt':30,'caption_font_pt':40,'section_font_pt':48,'sources':g.sources,
        'left_data_preservation':audit,'command_diagnostics':diagnostics,'poster_sha256':sha(target)}
    (out/'manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    (out/'quality_checks.json').write_text(json.dumps({**check_svg(target),
        'protected_pixel_changes':protected,'left_data_unchanged':True,'all_generated_text_declares_helvetica':True,
        'added_text_on_page':True,'text_collisions':[],'references_boxed':True},indent=2)+'\n')
    write_evolved_notes(out)
    print(f'Evolved B written to {target}')


def write_evolved_notes(out: Path):
    """Explain the different control tests without expanding the poster caption."""
    (out/'README.md').write_text('''# Selected B: field shape, sensitivity and commanded success

[Editable A0 poster](Bernstein2026_B_evolved.svg) · [Full-page preview](Bernstein2026_B_evolved.png)

This evolves the selected B. Transfer, the training-only predictor, the 168-run DG-16 architectural survey, and minimum-threshold field dominance versus exploration and recorded target hits remain. Section 2 is tightened to fit a new command row. The header and introduction are unchanged; all nine left-panel map, peak and trajectory data fingerprints are unchanged. A shared blue-circle **Option start** / gold-star **Goal hit** key labels the bottom-left trajectory panels. Goal hits mean internal DG events. References are boxed.

All generated plot and diagram labels now declare **Helvetica**, at 30 pt; conclusion captions remain 40 pt. Helvetica is not installed on this machine. Rendering and glyph measurement use the verified scalable **Nimbus Sans** substitute explicitly; the SVG keeps `Helvetica,Nimbus Sans` so Helvetica is used when available. This substitution and font path are recorded in [manifest.json](manifest.json). The original header/introduction fonts remain unchanged.

## Two different aspects of controllability

- **Action-probability TV** measures sensitivity to swapping the goal at the same policy state: $\\frac{1}{2}\\sum_a |p(a)-p'(a)|$. The saved training diagnostic rolls the goal across the minibatch and averages valid original-goal rows. It is invariant to a constant logit shift. It measures instantaneous probability change, not successful command execution.
- **Command lift** is executed success minus matched shuffled-command success, in percentage points. It tests target-specific internal DG outcomes over command trials. The shuffled comparator is retrospective, physical starts are not guaranteed identical, and ordered-pair coverage is incomplete.

The main command row shows all **12 models / four variants / three seeds**, DG 16 at 75,038,720 frames: First versus Hit DG policies, and Source versus Arrival credit. Every saved model has positive command lift, **+11.3 to +29.4 p.p.** Lines join matching training seeds; black bars are means. This is an absolute difference, distinct from section 2b's training hit-lift ratio.

TV is locally available for the **six overlapping Source/Arrival credit models**, from the 70M–74,973,180 training window: **0.029–0.408%** probability mass. Their frozen command lifts are **19.1–25.7 p.p.** Low instantaneous goal sensitivity can coexist with measurable command specificity. These diagnostics use different time windows and interventions. No TV values are saved for the CPD cohort used in the field-dominance panel; missing values are not filled in.

## Field dominance versus these measures

[Command-lift scatter](plots/dominance_command.svg) · [TV scatter](plots/dominance_tv.svg)

These are separate candidate figures, leaving the main poster readable. Replay mean-threshold dominance versus command lift has $\\rho=-0.31$ for DG policy and $\\rho=+0.77$ for credit assignment, with six points per family. Credit dominance versus TV gives $\\rho=-0.26$, also six points. Opposite command-lift associations do not support a general claim that dominant fields harm control. The poster's message remains **field dominance does not reliably order better behaviour**.

The command representations all use the same saved 10,001-observation replay panel. This **mean-threshold dominance** differs from B's **minimum-threshold dominance** in recent online CPD maps; these scores and protocols remain separate. DG capacity stays at 16. No causal claim or significance mark is added. [Continuous-field evidence](../../continuous_fields/README.md) documents sampling and protocol limitations.

## Verification and reuse

[Quality checks](quality_checks.json) verify protected pixels, data fingerprints, page bounds, Helvetica declarations and text collisions. [Plot checks](plots/quality_checks.json) verify native vectors and a 30 pt minimum. [Exact plotted coordinates](plots/plotted_points.csv), [TV joins](plots/action_tv_per_run.csv) and [diagnostic definitions/statistics](plots/command_diagnostics_statistics.json) retain the evidence. No new telemetry or cluster access was needed.

Regenerate from the repository root:

```bash
/home/xiaoxiong/miniforge3/envs/SF_git/bin/python 06_experiments/revise_poster_layout_20260929.py --evolve-b
```

**Reusable experience:** audit which control diagnostic is actually exported before choosing its cohort; raw-logit differences cannot supply missing probability TV. Exact model joins reuse the offline command evidence. Declare the requested font, verify the installed substitute, measure real glyph bounds and preserve data fingerprints when typography changes intentionally.
''')


REVIEW_VARIANTS = {
    'V1_dominance_TV':{'title':'Dominance + TV', 'field_plot':'review_dominance',
        'field_heading':'3b  Field dominance in four CA3-feedback variants',
        'field_caption':'Dominance does not order better behaviour here.',
        'field_note':'12 runs · DG 16 · 75M · architecture associations', 'keep_tv':True},
    'V2_concentration':{'title':'Concentration + command success', 'field_plot':'review_concentration',
        'field_heading':'3b  Spatial concentration in four CA3-feedback variants',
        'field_caption':'Higher C: less coverage; target association weak.',
        'field_note':'12 runs · DG 16 · 75M · coverage near ceiling', 'keep_tv':False},
    'V3_core_concentration':{'title':'Core designs + command success', 'field_plot':'review_core_shape',
        'field_heading':'3b  Core designs: exploration and map shape',
        'field_caption':'C15: more exploration, less concentrated DG maps.',
        'field_note':'9 models · 100M · training AUC / frozen-map concentration', 'keep_tv':False},
}


def review_exploration(g: Gallery):
    """Four compact panels expose mobility and the actual strict-field counts."""
    data=g.read(TABLE);data=data[data.condition.isin(COLORS)].copy()
    fields=g.read(FLAT/'mono_field_peak_counts.csv')
    fields=fields[fields.condition.isin(COLORS)].copy()
    data=data.merge(fields[['condition','seed','mono_units','dg_units']],on=['condition','seed'],validate='one_to_one')
    if len(data)!=9 or not data.dg_units.eq(16).all():raise ValueError('Core field-count cohort changed')
    if data.groupby('condition').mono_units.sum().to_dict()!={'C01':1,'C05':2,'C15':1}:
        raise ValueError('Strict-field count claim changed')
    fig,axes=scalar_axes(4)
    specs=[('coverage_auc_terminal','A  AUC\n(cells)',1),
           ('return_20_mobile','B  Mobile returns\n20 steps (%)',100),
           ('mobile_20_fraction','C  Mobile windows\n20 steps (%)',100),
           ('mono_units','D  Strict fields\n/ 48 DG units',1)]
    for ax,(metric,title,factor) in zip(axes,specs):
        if metric=='mono_units':
            ax.set_title(title);ax.axis('off')
            totals=data.groupby('condition',sort=True).mono_units.sum()
            for x,(condition,count) in zip((.18,.5,.82),totals.items()):
                ax.text(x,.70,f'{int(count)}/48',transform=ax.transAxes,ha='center',fontsize=30,color=COLORS[condition])
                ax.text(x,.02,condition,transform=ax.transAxes,ha='center',fontsize=30)
            aggregate=totals.rename('strict_field_count').reset_index().assign(seed=np.nan)
            g.record('review_exploration',title,'strict_field_count',aggregate,'strict_field_count',
                     protocol='frozen_10k_policy_probe',dg_units=16,checkpoint_frames=100_040_704,
                     sample_unit='sum over three training seeds',denominator_units=48)
            continue
        dots(ax,data,metric,tuple(COLORS),factor)
        for collection in ax.collections:
            collection.set_sizes([78]);collection.set_clip_on(False)
        ax.set(title=title,ylim=(0,100),yticks=[0,50,100])
        g.record('review_exploration',title,metric,data,metric,factor,dg_units=16,
                 checkpoint_frames=100_040_704,
                 protocol='last_10m_training_mean' if metric=='coverage_auc_terminal' else 'frozen_10k_policy_probe')
    g.finish(fig,Candidate('review_exploration','Exploration, mobility and strict fields',ROW_MM,'All review versions','','','',''))
    data.to_csv(g.out/'review_core_per_run.csv',index=False)
    return data


def review_transfer(g: Gallery):
    """Join the third arm without relabelling heldout results as package tests."""
    from analyze_poster_reward_auc import reward_series,mean_reward_auc
    from hpc_runs.intrmotiv_study import load_study
    data=g.read(TRANSFER/'reward_per_run.csv').copy()
    data['arm_display']=data.arm.map({'W_RAND_DG':'Rand','W_WORKER':'Package'})
    folder=ROOT/'06_experiments/data/worker_random_transfer_20260929/studies'
    declared={}
    for path in sorted(folder.glob('*.study.json')):
        g.sources[str(path.relative_to(ROOT))]=sha(path)
        spec=load_study(path)
        for run in spec.expand_runs():declared[run.name]=(run,spec)
    source_rows=[];backend=[]
    for architecture in ('D50','D51'):
        for seed in (42,1234,9999):
            run_name=f'CR5C_{architecture}_W_SOURCE_DG_S{seed}'
            run,spec=declared[run_name]
            if run.seed!=seed or run.factors['arm']!='W_SOURCE_DG':raise ValueError('Third arm study mismatch')
            path=ROOT/'06_experiments/data/poster_missing_analyses_20260926/reward_histories'/f'{run_name}.csv'
            g.sources[str(path.relative_to(ROOT))]=sha(path)
            frames,rewards=reward_series(path)
            if frames[-1]<.995*75_000_000:raise ValueError('Source-DG history incomplete')
            source_rows.append({'run_name':run_name,'architecture':architecture,'arm':'W_SOURCE_DG',
                'condition':'W_SOURCE_DG','arm_display':'DG only','seed':seed,
                'full_0_75m':mean_reward_auc(frames,rewards,75_000_000),
                'early_0_10m':mean_reward_auc(frames,rewards,10_000_000),
                'study_schema':'intrmotiv/study/v1','workflow_version':spec.declared_workflow_version,'study_sha256':spec.fingerprint})
            # Source-DG uses an existing TensorBoard export; the other two arms
            # use pinned W&B logs. Check their overlapping random controls rather
            # than silently treating the backends as byte-identical.
            random_path=path.with_name(f'CR5C_{architecture}_W_RAND_DG_S{seed}.csv')
            g.sources[str(random_path.relative_to(ROOT))]=sha(random_path)
            frames,rewards=reward_series(random_path)
            tb=mean_reward_auc(frames,rewards,75_000_000)
            wb=float(data[(data.architecture==architecture)&(data.seed==seed)&(data.arm=='W_RAND_DG')].full_0_75m.iloc[0])
            difference=abs(wb-tb)/tb
            if difference>.005:raise ValueError('Logging backend difference exceeds 0.5%; re-audit three-arm plot')
            backend.append({'architecture':architecture,'seed':seed,'tensorboard_random':tb,
                            'wandb_random':wb,'relative_backend_difference':difference})
    data=pd.concat([data,pd.DataFrame(source_rows)],ignore_index=True)
    if len(data)!=18 or data.duplicated(['architecture','arm','seed']).any():raise ValueError('Three-arm cohort changed')
    fig,axes=scalar_axes()
    statistics=[]
    arms=('W_RAND_DG','W_SOURCE_DG','W_WORKER');labels=('Rand','DG','Pkg')
    palette=('#777777','#D55E00','#0072B2')
    for ax,architecture in zip(axes,('D50','D51')):
        group=data[data.architecture.eq(architecture)]
        pivot=group.pivot(index='seed',columns='arm',values='full_0_75m').sort_index()
        if pivot.isna().any().any() or len(pivot)!=3:raise ValueError('Incomplete transfer seed triple')
        for offset,(_,row) in zip((-.13,0,.13),pivot.iterrows()):
            ax.plot(np.arange(3)+offset,row[list(arms)].to_numpy()*1000,color='#aaaaaa',lw=1,zorder=1)
        for index,arm in enumerate(arms):
            values=pivot[arm].to_numpy()*1000
            ax.scatter(index+np.array([-.13,0,.13]),values,s=78,color=palette[index],
                       marker=('o','s','^')[index],zorder=3)
            ax.plot([index-.2,index+.2],[values.mean()]*2,color='black',lw=2,zorder=4)
        ax.set(title=f'{architecture} Reward\n/ step × 10³',xticks=[0,1,2],xticklabels=labels,
               xlim=(-.5,2.5),ylim=(0,.6),yticks=[0,.3,.6]);ax.grid(axis='y',color='#eeeeee')
        source_wins=int((pivot.W_SOURCE_DG>pivot.W_RAND_DG).sum())
        package_wins=int((pivot.W_WORKER>pivot.W_RAND_DG).sum())
        mean=pivot.mean()
        early=group.groupby('arm').early_0_10m.mean()
        if source_wins!=0 or package_wins!=2 or not early.W_WORKER<early.W_RAND_DG:
            raise ValueError('Transfer-caption evidence changed')
        statistics.append({'architecture':architecture,'source_seed_pairs_won':source_wins,
            'package_seed_pairs_won':package_wins,'dg_only_relative_gain':float(mean.W_SOURCE_DG/mean.W_RAND_DG-1),
            'package_relative_gain':float(mean.W_WORKER/mean.W_RAND_DG-1),
            'package_early_relative_gain':float(early.W_WORKER/early.W_RAND_DG-1)})
        g.record('review_transfer',architecture,'0–75M mean reward',group,'full_0_75m',1000,
                 dg_units=64,protocol='frame-weighted logged reward during training',horizon_frames=75_000_000)
    g.finish(fig,Candidate('review_transfer','DG-only and package transfer',(191.5,76.2),'All review versions','','','',''))
    data.to_csv(g.out/'review_transfer_per_run.csv',index=False)
    report={'arms':statistics,'backend_crosscheck':backend,
        'label':'Package = W_WORKER; W_FULL is a different, DG-trainable study arm',
        'limitations':['One selected source checkpoint per architecture, reused across downstream seeds',
                       'Source pretraining is additional compute; worker and graph effects remain combined',
                       'Logged training reward; package heldout physical success unavailable']}
    (g.out/'review_transfer_statistics.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def review_survey(g: Gallery):
    """The same 168 observations, separated by family without fitted lines."""
    raw=g.read(SCATTER)
    data=raw[(raw.geometry_group=='legacy_19x19')&(raw.dg_units==16)&
             (raw.protocol=='online_latest_saved_window')].dropna(
                subset=['spatial_information','unique_peak_bins','prospective_success']).copy()
    if len(data)!=168 or data.run_name.duplicated().any():raise ValueError('Survey cohort changed')
    fig,axes=plt.subplots(1,3,figsize=(383/25.4,100/25.4),sharey=True)
    fig.subplots_adjust(left=.12,right=.98,top=.72,bottom=.48,wspace=.28)
    statistics=[]
    for ax,family in zip(axes,('ALL','CPD','DGP')):
        group=data if family=='ALL' else data[data.family.eq(family)]
        for name,rows in group.groupby('family'):
            ax.scatter(rows.spatial_information,rows.prospective_success*100,s=78,alpha=.8,
                       color=FAMILY_COLORS[name],marker=FAMILY_MARKERS[name],edgecolor='white',lw=.35)
        rho=float(group.spatial_information.corr(group.prospective_success,method='spearman'))
        displayed=0 if abs(rho)<.005 else rho
        title={'ALL':'Pooled spatial score','CPD':'CA3 feedback','DGP':'DG policy'}[family]
        ax.set(title=f'{title}\nn={len(group)} · ρ={displayed:.2f}',xlim=(0,.56),xticks=[0,.2,.4],
               ylim=(0,100),yticks=[0,50,100]);ax.grid(color='#eeeeee')
        ax.tick_params(axis='x',pad=14)
        statistics.append({'family':family,'n':len(group),'spearman_rho':rho,
                           'minimum_frames':int(group.frames.min()),'maximum_frames':int(group.frames.max())})
        g.record('review_survey',family,'spatial_information',group,'spatial_information',x_axis=True,dg_units=16)
        g.record('review_survey',family,'recorded target-event success',group,'prospective_success',100,y_axis=True,dg_units=16)
    axes[0].set_ylabel('Target-event\nsuccess (%)')
    # The first title names the shared horizontal metric; a repeated label in
    # the narrow gap between ticks and legend obscured the family key.
    handles=[Line2D([],[],marker=FAMILY_MARKERS[f],color=FAMILY_COLORS[f],ls='none',markersize=8,
                    label=FAMILY_NAMES[f]) for f in sorted(data.family.unique())]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.04),ncol=3,frameon=False,
               handletextpad=.35,columnspacing=.8)
    g.finish(fig,Candidate('review_survey','Pooled versus within-family associations',(383,100),'All review versions','','','',''))
    (g.out/'review_survey_statistics.json').write_text(json.dumps(statistics,indent=2)+'\n')
    return data


def review_fields(g: Gallery,core: pd.DataFrame):
    data=g.read(CONTINUOUS/'per_run.csv')
    cpd=data[data.family.eq('CPD')&data.protocol.eq('online_latest_saved_window')].copy()
    if len(cpd)!=12 or not cpd.frames.eq(75_005_952).all():raise ValueError('Continuous CPD cohort changed')
    statistics=[]
    for metric,label in [('dominance','Field dominance'),('concentration','Concentration C')]:
        key='review_'+metric
        fig,axes=plt.subplots(1,2,figsize=tuple(v/25.4 for v in ROW_MM))
        fig.subplots_adjust(left=.16,right=.98,bottom=.43,top=.77,wspace=.50)
        for ax,outcome,title in zip(axes,('exploration_coverage','prospective_success'),('Exploration','Target events')):
            ax.scatter(cpd[metric],cpd[outcome]*100,s=78,color='#0072B2',alpha=.78,edgecolor='white',lw=.5)
            rho=float(cpd[metric].corr(cpd[outcome],method='spearman'))
            ax.set(title=f'{title} · ρ={rho:.2f}',xlabel=label,
                   ylabel='Visited\n(%)' if outcome=='exploration_coverage' else 'Success\n(%)',
                   xlim=(0,1),xticks=[0,.5,1],ylim=(0,100),yticks=[0,50,100]);ax.grid(color='#eeeeee')
            ax.tick_params(axis='x',pad=14)
            statistics.append({'metric':metric,'outcome':outcome,'n':12,'variants':4,'spearman_rho':rho})
            g.record(key,title,metric,cpd,metric,x_axis=True,dg_units=16,protocol='online_latest_saved_window')
            g.record(key,title,outcome,cpd,outcome,100,y_axis=True,dg_units=16,protocol='online_latest_saved_window')
        g.finish(fig,Candidate(key,label+' and behaviour',ROW_MM,'Review variant','','','',''))
    frozen=data[data.family.eq('Corrected core')&data.code.isin(COLORS)&
                data.protocol.eq('frozen_latest_archived_probe')].copy()
    if len(frozen)!=9 or not frozen.frames.eq(100_040_704).all():raise ValueError('Core concentration cohort changed')
    frozen['condition']=frozen.code
    joined=core[['condition','seed','coverage_auc_terminal']].merge(frozen,on=['condition','seed'],
        validate='one_to_one',suffixes=('_training',''))
    joined['coverage_auc_terminal']=joined.coverage_auc_terminal_training
    fig,ax=plt.subplots(figsize=(191.5/25.4,76.2/25.4),layout='constrained')
    for index,condition in enumerate(COLORS):
        rows=joined[joined.condition.eq(condition)]
        ax.scatter(rows.concentration,rows.coverage_auc_terminal,s=78,color=COLORS[condition],
                   marker=('o','s','^')[index],label=condition)
    ax.set(title='Same 9 core models',xlabel='Concentration C',ylabel='AUC\n(cells)',
           xlim=(.65,1),xticks=[.7,.85,1],ylim=(0,100),yticks=[0,50,100])
    ax.legend(frameon=False,loc='upper left',handlelength=.7,handletextpad=.35,borderpad=.2,labelspacing=.2)
    ax.grid(color='#eeeeee')
    for condition,rows in joined.groupby('condition'):
        g.record('review_core_shape',condition,'concentration',rows,'concentration',x_axis=True,
                 protocol='frozen_latest_archived_probe',dg_units=16,checkpoint_frames=100_040_704)
        g.record('review_core_shape',condition,'coverage_auc_terminal',rows,'coverage_auc_terminal',y_axis=True,
                 protocol='last_10m_training_mean',dg_units=16,checkpoint_frames=100_040_704)
    g.finish(fig,Candidate('review_core_shape','Core exploration and map shape',(191.5,76.2),'Core-focused variation','','','',''))
    joined.to_csv(g.out/'review_core_shape_per_run.csv',index=False)
    statistics.append({'metric':'core_concentration','means':joined.groupby('condition').concentration.mean().to_dict(),
                       'x_axis_detail_range':[.65,1],'different_observation_protocols':True})
    (g.out/'review_field_statistics.json').write_text(json.dumps(statistics,indent=2)+'\n')


def review_package_diagram(layer,x,y):
    for index,(name,labels,color) in enumerate([
        ('RAND',('Random','Fresh','Empty'),'#777777'),
        ('DG ONLY',('DG 64','Fresh','Empty'),'#D55E00'),
        ('PACKAGE',('DG 64','Worker','Graph'),'#0072B2')]):
        yy=y+index*27
        text(layer,'package-'+name+'-name',x,yy+8,[name],30,bold=True,color=color)
        for i,(offset,width,label) in enumerate(zip((0,60,122),(50,52,56),labels)):
            diagram_box(layer,f'package-{index}-{i}',x+offset,yy+11,width,label,color,height=15)


def concentration_key(layer,x,y):
    """Mathematical endpoints, visibly labelled schematic; no invented data."""
    text(layer,'concentration-schematic',x,y+9,['Concentration C · schematic'],30,bold=True)
    for index in range(2):
        xx=x+12+index*85
        for row in range(5):
            for column in range(5):
                color='#0072B2' if index==0 or (row,column)==(2,2) else '#eeeeee'
                ET.SubElement(layer,f'{{{SVG}}}rect',{'id':f'concentration-key-{index}-{row}-{column}',
                    'x':str(xx+column*7),'y':str(y+16+row*7),'width':'6.7','height':'6.7','fill':color})
        text(layer,f'concentration-endpoint-{index}',xx-10,y+65,
             ['Uniform: C=0' if index==0 else 'One bin: C=1'],30)


def compose_review(root,g: Gallery,key,reference):
    remove(root,find(root,'revised-right-B_transfer_predictor_dominance'))
    layer=ET.SubElement(root,f'{{{SVG}}}g',{'id':'review-right-'+key,
        f'{{{INK}}}groupmode':'layer',f'{{{INK}}}label':'Review · '+REVIEW_VARIANTS[key]['title']})
    x=442.4;item=REVIEW_VARIANTS[key]
    section(layer,2,'Explore, transfer and predict',x,220)
    text(layer,'subsection-2a',x,244,['2a  Exploration, mobility and strict fields'],36,bold=True,color='#253B57')
    text(layer,'landmark-key',x,263,['Landmark = internal DG event; spatial tuning tested later.'],30)
    embed(layer,g.out/'review_exploration.svg','review-exploration',x,272,383)
    text(layer,'coverage-gain',x,363,['C01/C15: 1/48 strict fields; C15: 2.1× AUC.'],40,bold=True)
    text(layer,'exploration-protocol',x,377,['A: training mean · B–D: frozen 10k probes · 3 seeds'],30)
    text(layer,'subsection-2b',x,398,['2b  Does target identity change the policy?'],36,bold=True,color='#253B57')
    embed(layer,g.out/'goal_specificity.svg','goal-specificity',x,409,191.5)
    control_diagram(layer,645.4,409)
    text(layer,'goal-control-caption',x,503,['C15: weak signal; hit lift below 1 in all 3 seeds.'],40)
    text(layer,'subsection-2c',x,526,['2c  DG-only versus control-package transfer'],36,bold=True,color='#253B57')
    embed(layer,g.out/'review_transfer.svg','transfer-result',x,537,191.5)
    review_package_diagram(layer,645.4,537)
    text(layer,'transfer-protocol',x,622,['DG 64 · 75M · Pkg: 2/3 pairs each'],30)
    text(layer,'transfer-caption',x,635,['Package: +6% / +16% mean; no early head start.'],40)
    text(layer,'subsection-2d',x,654,['2d  Adding a goal-conditioned next-event predictor'],36,bold=True,color='#253B57')
    embed(layer,g.out/'predictor.svg','predictor-result',x,665,191.5)
    predictor_diagram(layer,645.4,665)
    text(layer,'predictor-caption',x,759,['More revisits, less coverage in all 3 seed pairs.'],40)
    section(layer,3,'Field shape and internal control',x,780)
    text(layer,'survey-protocol',x,805,['3a  Pooled vs within-family · DG 16 · 25–150M'],36,bold=True,color='#253B57')
    embed(layer,g.out/'review_survey.svg','survey-main',x,815,383)
    text(layer,'subsection-3b',x,934,[item['field_heading']],36,bold=True,color='#253B57')
    embed(layer,g.out/(item['field_plot']+'.svg'),'field-behavior',x,945,
          191.5 if key=='V3_core_concentration' else 383)
    if key=='V3_core_concentration':concentration_key(layer,645.4,945)
    text(layer,'field-message',x,1037,[item['field_caption']],40,bold=True)
    text(layer,'field-protocol',x,1053,[item['field_note']],30)
    text(layer,'subsection-3c',x,1074,
         ['3c  Goal sensitivity and internal target specificity' if item['keep_tv'] else
          '3c  Executed commands show internal target specificity'],36,bold=True,color='#253B57')
    embed(layer,g.out/'command_lift.svg','command-lift',x,1085,191.5)
    if item['keep_tv']:
        embed(layer,g.out/'action_tv.svg','action-tv',633.9,1085,191.5)
        text(layer,'command-message',x,1172,['12 positive lifts · internal DG events; TV: 6 models.'],40)
    else:
        text(layer,'command-positive',645.4,1098,['Positive lift in all','12 evaluated models'],40,step=17,bold=True)
        text(layer,'command-definition',645.4,1134,['Executed − matched shuffled','success, in percentage points'],30,step=12)
        text(layer,'command-caveat',645.4,1160,['Internal DG events; arrival unverified'],30)
    ET.SubElement(layer,f'{{{SVG}}}rect',{'id':'references-box','x':'441.4','y':'1176','width':'385',
        'height':'12','rx':'1.5','fill':'#F4F6F8','stroke':'#253B57','stroke-width':'.6'})
    text(layer,'references',445.4,1186,[reference],30)
    helvetica_text(layer)
    return layer


def review_quality(out: Path,source: Path,source_png: Path,changed_key_id: str):
    """Compare the attached author-edited source, with one explicit key correction."""
    from PIL import Image
    prior=ET.parse(source).getroot();prior_bounds=query_bounds(source)
    original=np.asarray(Image.open(source_png).convert('RGBA'));checks=[]
    for key in REVIEW_VARIANTS:
        path=out/f'Bernstein2026_{key}.svg';root=ET.parse(path).getroot();bounds=query_bounds(path)
        actual=np.asarray(Image.open(path.with_suffix('.png')).convert('RGBA'))
        left_end=round(431/841*1800)
        difference=np.any(original[:,:left_end]!=actual[:,:left_end],axis=2)
        # Exempt only the union of old/new glyph bounds for the corrected map key.
        for box in [prior_bounds[changed_key_id],bounds[changed_key_id]]:
            x,y,w,h=box
            x0,y0,x1,y1=[round(v/841*1800) for v in (x-1,y-1,x+w+1,y+h+1)]
            difference[max(y0,0):y1,max(x0,0):min(x1,left_end)]=False
        if difference.any():raise ValueError(f'Other attached left/header pixels changed: {key}')
        header_end=round(193/841*1800)
        if np.any(original[:header_end]!=actual[:header_end]):raise ValueError('Attached header changed')
        for ids in LEFT_FIGURES.values():
            for ident in ids:
                if preserved_data(find(prior,ident))!=preserved_data(find(root,ident)):
                    raise ValueError('Attached plotted left data changed')
        for condition in COLORS:
            ident=condition+'-subheading'
            if ET.tostring(find(prior,ident))!=ET.tostring(find(root,ident)):
                raise ValueError('Rejected suggestion 1 changed a condition label')
        layer=find(root,'review-right-'+key)
        labels=[e for e in layer.iter(f'{{{SVG}}}text') if e.get('id') in bounds]
        clipped=[];collisions=[]
        for i,e in enumerate(labels):
            a=bounds[e.get('id')]
            if a[0]<0 or a[1]<0 or a[0]+a[2]>841 or a[1]+a[3]>1189:clipped.append(e.get('id'))
            for other in labels[i+1:]:
                b=bounds[other.get('id')]
                overlap=np.minimum(a[:2]+a[2:],b[:2]+b[2:])-np.maximum(a[:2],b[:2])
                if (overlap>.3).all():collisions.append([e.get('id'),other.get('id')])
        if clipped or collisions:raise ValueError(f'{key}: clipped={clipped}; overlaps={collisions}')
        checks.append({**check_svg(path),'source_header_pixel_changes':0,'other_left_pixel_changes':0,
            'condition_labels_unchanged':True,'left_plotted_data_unchanged':True,'right_text_on_page':True,
            'right_text_collisions':[],'corrected_map_key':changed_key_id,'svg_sha256':sha(path)})
    (out/'quality_checks.json').write_text(json.dumps(checks,indent=2)+'\n')


def review_overview(out: Path):
    root=ET.Element(f'{{{SVG}}}svg',{'width':'2540','height':'1320','viewBox':'0 0 2540 1320'})
    ET.SubElement(root,f'{{{SVG}}}rect',{'width':'2540','height':'1320','fill':'white'})
    for i,(key,item) in enumerate(REVIEW_VARIANTS.items()):
        x=20+i*840
        label=ET.SubElement(root,f'{{{SVG}}}text',{'x':str(x),'y':'50',
            'style':'font-family:Helvetica,Nimbus Sans;font-size:30px;font-weight:bold'})
        label.text=key[:2]+' · '+item['title']
        png=out/f'Bernstein2026_{key}.png'
        ET.SubElement(root,f'{{{SVG}}}image',{'x':str(x),'y':'90','width':'800','height':str(800*1189/841),
            'href':'data:image/png;base64,'+base64.b64encode(png.read_bytes()).decode()})
    target=out/'comparison_overview.svg';export_svg(root,target);render(target,2540)


def review_critique(source: Path,critique: Path,out: Path):
    """Review variations from the attached SVG, not a stale repository layout."""
    out.mkdir(parents=True,exist_ok=True);plots=out/'plots';plots.mkdir(exist_ok=True)
    pinned=out/'input_poster.svg';pinned.write_bytes(source.read_bytes())
    (out/'input_critique.txt').write_bytes(critique.read_bytes())
    source_digest=sha(source)
    font=style('Helvetica',fallback_family='Nimbus Sans');g=Gallery(plots)
    for path in (pinned,Path(__file__),Path(__file__).with_name('compose_a0_poster_20260929.py'),
                 Path(__file__).with_name('render_poster_candidate_plots_20260929.py'),
                 Path(__file__).with_name('analyze_poster_reward_auc.py'),
                 ROOT/'hpc_runs/intrmotiv_study/field_concentration.py'):
        g.sources[str(path.relative_to(ROOT))]=sha(path)
    inserted_results(g);core=review_exploration(g);transfer=review_transfer(g)
    review_survey(g);review_fields(g,core);command_diagnostics(g)
    # The exact core models have no exported TV. Keep literal logits and an
    # unmistakable no-advantage reference rather than fabricating TV from means.
    for path in g.figures:
        figure=ET.parse(path).getroot();helvetica_text(figure)
        # Matplotlib normally puts IDs on enclosing groups, not editable text.
        # Give each text its own ID so Inkscape can audit actual glyph bounds.
        for index,label in enumerate(figure.iter(f'{{{SVG}}}text')):
            if not label.get('id'):label.set('id',f'{path.stem}-glyph-{index}')
        if path.stem=='goal_specificity':
            for e in figure.iter(f'{{{SVG}}}text'):
                if e.text=='Action-score':e.text='Mean |Δlogit|'
                elif e.text=='change':e.text='goal swap'
            # Matplotlib's baseline line is a named dashed line2d path.
            for e in figure.iter(f'{{{SVG}}}path'):
                css=e.get('style','')
                if 'stroke-dasharray' in css and 'stroke: #777777' in css:
                    e.set('style',css.replace('stroke-width: 1.5','stroke-width: 2.5').replace('#777777','#253B57'))
        elif path.stem=='predictor':
            for e in figure.iter(f'{{{SVG}}}text'):
                if e.text=='Return (%)':e.text='Mobile returns'
                elif e.text=='20 decisions':e.text='20 steps (%)'
        export_svg(figure,path)
    root=ET.parse(pinned).getroot()
    key=next(e for e in root.iter(f'{{{SVG}}}text') if ''.join(e.itertext()).strip()=='DG id : spatial information')
    key_id=key.get('id')
    for span in key.iter(f'{{{SVG}}}tspan'):span.text='DG id : activity maximum'
    reference=''.join(find(root,'references').itertext()).strip()
    for variant in REVIEW_VARIANTS:
        actual=copy.deepcopy(root);compose_review(actual,g,variant,reference)
        path=out/f'Bernstein2026_{variant}.svg';export_svg(actual,path);render(path)
    source_png=out/'input_poster.png';render(pinned)
    review_quality(out,pinned,source_png,key_id)
    verify_and_render(g,pinned,source_digest)
    pd.DataFrame(g.points).to_csv(plots/'plotted_points.csv',index=False)
    review_overview(out)
    (out/'manifest.json').write_text(json.dumps({'schema':'intrmotiv/poster-critique-review/v1',
        'workflow_version':WORKFLOW_VERSION,'study_schema':'intrmotiv/study/v1',
        'source_attachment':str(source),'source_sha256':source_digest,'critique_sha256':sha(critique),
        'rejected_suggestion':1,'font_requested':'Helvetica','font_rendered':'Nimbus Sans','font_path':font,
        'plot_font_pt':30,'caption_font_pt':40,'variants':REVIEW_VARIANTS,
        'sources':g.sources,'transfer':transfer,'map_key_correction':key_id},indent=2,ensure_ascii=False)+'\n')
    if sha(source)!=source_digest:raise ValueError('Source attachment changed')
    write_review_notes(out)
    print(f'Three critique-review versions written to {out}')


def write_review_notes(out: Path):
    # Kept separate from data/geometry code so the assessment can be reviewed
    # without inspecting SVG generation mechanics.
    (out/'README.md').write_text('''# Three critique-review versions from the attached B

![Comparison](comparison_overview.png)

| Version | Editable A0 poster | Main choice |
| --- | --- | --- |
| V1 | [Dominance + TV](Bernstein2026_V1_dominance_TV.svg) | Closest to B: literal dominance and the six-model TV diagnostic |
| V2 | [Concentration + command success](Bernstein2026_V2_concentration.svg) | Actual spatial concentration in 12 CPD models; positive commands without TV |
| V3 | [Core designs + command success](Bernstein2026_V3_core_concentration.svg) | Recommended: the main nine C01/C05/C15 models link exploration and concentration |

All versions start from [the attached edited SVG](input_poster.svg), preserving its author/affiliation edits, introduction, map images, peak locations and trajectory geometry. **Suggestion 1 is not implemented:** all C01/C05/C15 headings remain exactly as attached. The single left annotation correction is the map key: displayed values are activity maxima, not spatial-information scores. Source values and scores are in [the individual-scale table](../../../../06_experiments/results/A0_poster_analysis_20260926/flat_goal_comparison/place_field_individual_scales.csv). No map values are changed.

The versions retain Helvetica declarations, verified Nimbus Sans rendering, 30 pt plot labels, 40 pt result captions and boxed references. The opening right-column key defines a landmark as an internal DG event, with spatial tuning evaluated post hoc. The schematic in V3 illustrates mathematical concentration endpoints and contains no fabricated observations. The core concentration x axis explicitly spans 0.65–1; all observations fit. Other scatter axes retain full 0–1 measures and 0–100% outcomes.

## Evaluation of the critique

| Suggestion | Assessment and implementation |
| --- | --- |
| 1: replace core labels | Rejected by the user; left untouched. Comparisons remain descriptive and captions do not attribute differences to a single factor. |
| 2: show strict-field counts | Agree. Four panels now show coverage AUC, 20-decision conditional returns, mobility and strict fields. The counts 1/48, 2/48 and 1/48 are visible. Mobility exposes C05's roughly 20% mobile-window fraction. Training AUC and frozen-probe measurements are explicitly separated. |
| 2b: distinguish policy diagnostics from reaching | Agree. Literal mean absolute logit change stays because TV is unavailable for these exact core models. The hit-lift reference is stronger and C15's below-one result is stated for all three seeds. |
| 3: compare DG-only and package transfer | Agree for these selected sources. All three arms are shown using logged reward over the same 75M horizon. The caption reports a modest package mean advantage and the absence of an early head start. |
| 4: rename prediction | Agree with neutral wording: adding a goal-conditioned next-event predictor. The observed result remains more revisits and less probe coverage in all three seed pairs. Familiar-cycle stabilization remains a hypothesis. |
| 5: show within-family associations | Strongly agree. The same 168-run cohort appears as pooled, CA3-feedback and DG-policy panels. Rank correlations are approximately 0.58, 0.09 and 0.00. No fitted line or new replication is implied. Target-event success is explicitly internal. |
| 6: dominance versus concentration | Agree. V1 correctly names dominance; V2 uses actual concentration; V3 ties actual concentration to the core designs. None manipulates field shape or establishes that fields cause worse control. |
| 7: foreground positive command lift | Agree. The existing 12-model seed-paired result remains; V2/V3 replace TV with a plain definition and the internal-event caveat. V1 keeps TV as a distinct six-model diagnostic. |

## Where the critique goes too far

“DG representations alone do not transfer useful control” is too general. The supported statement is **DG-only transfer collects less training reward than random DG in these six seed pairs, while the learned DG–worker–graph package has a modest, seed-dependent mean advantage**. Worker and graph contributions remain combined. Package physical heldout success is unavailable, source pretraining adds compute, and one selected source checkpoint per architecture is reused across downstream seeds.

`W_FULL` is already a distinct study arm with trainable DG. The plotted package is `W_WORKER`, whose DG remains frozen. Therefore its display name is **Package**, not a renamed `W_FULL`. Random and DG-only arms both have fresh workers and empty graphs; Package transfers worker and graph as well. All use DG 64 and a 75M downstream budget. The old TensorBoard DG-only export and newer W&B arms share the canonical interval integrator; the overlapping random controls agree within 0.12% in full-horizon reward. Exact backend differences are recorded, not discarded.

“Concentrated fields do not help” is also too strong as a causal conclusion. The 12-model CPD concentration rank associations are approximately $\\rho=-0.77$ with coverage and $\\rho=-0.24$ with recorded target-event success, but coverage is near its ceiling and target counters accumulate history. Frozen CPD probes do not reproduce the online coverage ordering. The core concentration means are C01 0.925, C05 0.918 and C15 0.877, from frozen maps; their training AUC is a separate protocol. The models show dissociation, not equivalence or an intervention on field shape.

The critique calls 2a/2d the strongest results; this is reasonable at poster scale, with the limits above. The predictor comparison matches age, environment and probe length, not identical stochastic trajectories or starts. No significance claim is added from three seeds.

## Evidence and verification

[Quality checks](quality_checks.json) compare renders against the supplied source, verify unchanged condition labels and all nine left-data fingerprints, and reject page clipping or text collisions. [Plot checks](plots/quality_checks.json), [plotted points](plots/plotted_points.csv), [three-arm transfer](plots/review_transfer_statistics.json), [within-family statistics](plots/review_survey_statistics.json), [field statistics](plots/review_field_statistics.json) and [manifest](manifest.json) retain exact definitions and source hashes. No training, SSH or new telemetry was used.

Regenerate with the pinned attachment and critique:

```bash
/home/xiaoxiong/miniforge3/envs/SF_git/bin/python 06_experiments/revise_poster_layout_20260929.py --review-critique --source 05_plans/poster_20260929/layout_versions/critique_variations/input_poster.svg --critique 05_plans/poster_20260929/layout_versions/critique_variations/input_critique.txt
```

**Reusable experience:** always preserve the latest attached author edits; distinguish a user-rejected suggestion from unrelated factual corrections. Compare exact matched cohorts and integrate all transfer histories with the canonical reward convention. Check existing study arm names before choosing display labels. Keep concentration, connected-component dominance, instantaneous policy sensitivity and long-horizon command specificity separate, and inspect actual glyph bounds after every compact layout change.
''')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--evolve-b',action='store_true',help='Evolve the user-selected B with command diagnostics and Helvetica')
    parser.add_argument('--review-critique',action='store_true',help='Produce review variations from the explicitly supplied edited poster')
    parser.add_argument('--critique',type=Path)
    args=parser.parse_args()
    if args.review_critique:
        if args.critique is None:parser.error('--review-critique requires --critique')
        review_critique(args.source,args.critique,args.out or PREVIOUS/'critique_variations')
        return
    if args.evolve_b:
        evolve_b(args.out or PREVIOUS/'evolved_B')
        return
    args.out=args.out or DEFAULT_OUT
    args.out.mkdir(parents=True,exist_ok=True)
    plots=args.out/'plots';plots.mkdir(exist_ok=True)
    font=style();g=Gallery(plots)
    for path in (Path(__file__),Path(__file__).with_name('compose_a0_poster_20260929.py'),
                 Path(__file__).with_name('render_poster_candidate_plots_20260929.py'),
                 SCATTER.with_name('analysis_metadata.json')):
        g.sources[str(path.relative_to(ROOT))]=sha(path)
    digest=sha(args.source)
    pinned=args.out/'input_layout.svg'
    if args.source.resolve()!=pinned.resolve():pinned.write_bytes(args.source.read_bytes())
    for _,namespace in ET.iterparse(args.source,events=['start-ns']):
        prefix,uri=namespace
        if prefix!='svg':ET.register_namespace(prefix,uri)
    source=ET.parse(args.source).getroot()
    if source.get('viewBox')!='0 0 841 1189':raise ValueError('Expected supplied A0 page')
    bounds=query_bounds(args.source)
    right=copy.deepcopy(find(source,'poster-right-20260929'))
    # Preserve the original header and introductory text/diagram, including
    # off-page drafts. Only the explicit caption IDs below may be edited.
    protected={e.get('id'):ET.tostring(e) for e in source.iter() if e.get('id') and
               e.tag in {f'{{{SVG}}}text',f'{{{SVG}}}image'} and
               e.get('id') not in LEFT_CAPTIONS and e.get('id') in bounds and
               bounds[e.get('id')][0]>=0 and bounds[e.get('id')][1]<720}
    # Remove right-column elements from the protection set; retain introduction.
    right_ids={e.get('id') for e in right.iter()}
    protected={k:v for k,v in protected.items() if k not in right_ids}
    data=survey(g);inserted_results(g);field_results(g,data);predictor_role(g,args.out)
    common=copy.deepcopy(source)
    left_audit=left_typography(common,bounds,g)
    results=[]
    for key in VARIANTS:
        root=copy.deepcopy(common);compose_right(root,right,g,key)
        for ident,original in protected.items():
            if ET.tostring(find(root,ident))!=original:raise ValueError(f'Protected introduction changed: {ident}')
        path=args.out/f'Bernstein2026_{key}.svg';export_svg(root,path)
        result=check_svg(path);render(path)
        results.append({**result,'variant':key,'protected_objects_unchanged':len(protected),
                        'width_mm':841,'height_mm':1189,'source_sha256':digest})
    for path in g.figures:
        check_svg(path);render(path,round(float(ET.parse(path).getroot().get('width').removesuffix('mm'))/25.4*150))
    pd.DataFrame(g.points).to_csv(plots/'plotted_points.csv',index=False)
    report={'schema':'intrmotiv/poster-layout-alternatives/v2','workflow_version':WORKFLOW_VERSION,
            'study_schema':'intrmotiv/study/v1','source':str(args.source),'source_sha256':digest,
            'previous_left_reference_sha256':sha(PREVIOUS/'Bernstein2026_A_system_transfer.svg'),
            'font_path':font,'plot_font_pt':30,'body_font_pt':40,'section_font_pt':48,'subsection_font_pt':36,
            'sources':g.sources,'pinned_source':'input_layout.svg','variants':VARIANTS,'left_data_preservation':left_audit,'quality':results,
            'study_metadata':'Original study SHA-256 values retained in point tables; no study or telemetry changes.'}
    (args.out/'manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    if sha(args.source)!=digest:raise ValueError('Supplied SVG changed')
    visual_checks(args.out,args.source)
    baseline=PREVIOUS/'Bernstein2026_A_system_transfer.svg'
    verify_and_render(g,baseline,sha(baseline))
    overview(args.out)
    write_notes(args.out)
    print(f'Two A0 alternatives written to {args.out}')


if __name__=='__main__':
    main()
