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
from render_poster_candidate_plots_20260929 import Candidate, Gallery, ROW_MM, paired, scalar_axes
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
        'field_caption':'Single fields do not reliably track better behaviour.',
        'field_protocol':'CA3 feedback · 27 variants × 3 seeds · DG 16 · 75M',
        'claim':'Single-field fraction has no positive exploration/control ordering in the 81-run CA3-feedback family.',
        'risk':'Online coverage is near its ceiling. Maps use a recent behavior window; target hits are historical internal events. '
               'Architectural association does not establish that monofields have no causal benefit.',
    },
    'B_transfer_predictor_dominance':{
        'title':'Continuous field dominance', 'recommended':False,
        'field_plot':'dominance_behavior',
        'field_caption':'Field dominance does not imply better behaviour.',
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


def field_results(g: Gallery, survey_data: pd.DataFrame):
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


def survey(g: Gallery):
    raw=g.read(SCATTER)
    data=raw[(raw.geometry_group=='legacy_19x19')&(raw.protocol=='online_latest_saved_window')&
             (raw.dg_units==16)].dropna(subset=['spatial_information','unique_peak_bins','prospective_success']).copy()
    if len(data)!=168 or data.condition.nunique()!=56 or data.run_name.duplicated().any():
        raise ValueError('The original fixed-DG-16 survey cohort changed')
    data['condition_original']=data.condition
    data['condition']=data.family
    statistics=[]
    fig,axes=plt.subplots(1,2,figsize=(383/25.4,130/25.4),sharey=True)
    fig.subplots_adjust(left=.12,right=.975,top=.88,bottom=.43,wspace=.28)
    for ax,x,label in zip(axes,('spatial_information','unique_peak_bins'),('Spatial score','Distinct peak bins')):
        rho=scatter_panel(ax,data,x,label,label)
        # Titles already name the horizontal measures. Removing the duplicate
        # labels leaves room for the six-family key at the original type size.
        ax.set_xlabel('')
        ax.text(.04,.97,f'ρ={rho:.2f}',transform=ax.transAxes,va='top')
        statistics.append({'panel':'main_'+x,'sample_unit':'run','n':len(data),'rho':rho})
        g.record('survey_main',x,'recorded DG target-hit fraction',data,'prospective_success',100,
                 dg_units=16,protocol='historical counters at latest saved online spatial window')
    axes[0].set_ylabel('DG hits (%)')
    handles=[Line2D([],[],marker=FAMILY_MARKERS[f],color=FAMILY_COLORS[f],ls='none',markersize=8,
                    label=FAMILY_NAMES[f]) for f in sorted(data.family.unique())]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.015),ncol=3,frameon=False,
               handletextpad=.4,columnspacing=.9)
    g.finish(fig,Candidate('survey_main','Current section 3, compact layout',(383,130),'All versions','','','',''))
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
         ['Eligible units · recent maps / historical hits · associations, not causal effects'],30)
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
        results.append({'variant':key,'protected_pixel_changes':regions,
                        'previous_left_column_pixel_changes':left_changes,'added_text_on_page':True,
                        'peak_labels_do_not_overlap':True,'svg_sha256':sha(path)})
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
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
    report={'schema':'intrmotiv/poster-layout-alternatives/v2','source':str(args.source),'source_sha256':digest,
            'font_path':font,'plot_font_pt':30,'body_font_pt':40,'section_font_pt':48,'subsection_font_pt':36,
            'sources':g.sources,'pinned_source':'input_layout.svg','variants':VARIANTS,'left_data_preservation':left_audit,'quality':results,
            'study_metadata':'Original study SHA-256 values retained in point tables; no study or telemetry changes.'}
    (args.out/'manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    if sha(args.source)!=digest:raise ValueError('Supplied SVG changed')
    visual_checks(args.out,args.source)
    overview(args.out)
    write_notes(args.out)
    print(f'Two A0 alternatives written to {args.out}')


if __name__=='__main__':
    main()
