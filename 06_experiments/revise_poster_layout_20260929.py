"""Build three reviewable A0 alternatives from the user's revised Inkscape SVG.

The supplied layout is immutable. Retain the header, introduction, actual map
images, every trajectory vertex, and all peak markers. Edit only identified
section labels and bottom-left plot typography; compose the right column with
scoped imports using the existing A0 figure helpers. Existing pinned studies
and measurements remain authoritative; no telemetry or training is launched.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
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
from render_poster_candidate_plots_20260929 import Candidate, Gallery, paired, scalar_axes
from matplotlib.lines import Line2D

SOURCE = Path('/home/xiaoxiong/.codex/attachments/56e5e916-3ac9-456f-9554-2f91aaab911e/Bernstein2026_IntrMotiv_layout.svg')
DEFAULT_OUT = ROOT/'05_plans/poster_20260929/layout_versions'
FLAT = ROOT/'06_experiments/results/A0_poster_analysis_20260926/flat_goal_comparison'
OLD_CANDIDATES = ROOT/'05_plans/poster_20260929/candidate_plots'
TRANSFER = ROOT/'05_plans/poster_20260929/worker_random_transfer'
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
    'A_system_transfer':{
        'title':'System transfer', 'recommended':True, 'candidate':'transfer_compact',
        'insert_title':'2c  Transfer to a rewarded task', 'extra':'within_families',
        'side':['WORKER: learned DG,','worker and graph.','RAND_DG: random DG,','fresh worker,','empty graph.'],
        'insert_caption':['Mean reward: +6% D50; +16% D51.','Two of three seed pairs win in each.'],
        'insert_protocol':'75M downstream frames · DG 64 · one source per architecture',
        'survey_subtitle':'3b  Does the trend hold within a family?',
        'survey_caption':['More peak bins do not imply more target hits.','Score–hit trends weaken within both families.','Historical hit counters; recent spatial windows.'],
        'claim':'Joint DG–worker–graph transfer improves average adaptation reward, with seed-dependent outcomes.',
        'takehome':'Broader exploration, with seed-dependent system transfer.',
        'question':'Which interactions make a learned code reusable by its worker and graph?',
        'risk':'Reward gains are modest and heterogeneous. Two of three paired seeds favor WORKER in each '
               'architecture; there is no matched WORKER heldout evaluation. This tests a transferred '
               'package, not a DG-specific effect or equal total pretraining compute.',
    },
    'B_predictor_and_graph':{
        'title':'Predictor and graph', 'recommended':False, 'candidate':'predictor',
        'insert_title':'2c  A predictor can accompany more looping', 'extra':'graph_grouping',
        'side':['Adding a goal','predictor increases','20-step returns','and reduces','visited coverage.'],
        'insert_caption':['Both directions hold in all three seed pairs.','Frozen 10k probes at the same 75M age.'],
        'insert_protocol':'DG 16 · already goal-conditioned · matched age, not identical paths',
        'survey_subtitle':'3b  Connectivity depends on the grouping',
        'survey_caption':['Connectivity–hit ordering changes by family.','Pooled ρ=−0.55; CA3 feedback ρ=0.67.','Stored connectivity does not verify executable edges.'],
        'claim':'Adding a goal predictor accompanies more short returns and less visited coverage in a matched three-seed contrast.',
        'takehome':'Exploration and internal-model diagnostics can diverge.',
        'question':'Can predicting familiar transitions stabilize cycles instead of discovery?',
        'risk':'The predictor result comes from one architectural background and stochastic frozen '
               'probes. Graph correlations are descriptive, with heterogeneous outcome rules and '
               'ages across families; they do not identify a causal failure of graph connectivity.',
    },
    'C_fields_and_variant_means':{
        'title':'Fields and variant means', 'recommended':False, 'candidate':'compact_fields',
        'insert_title':'2c  Compact fields remain uncommon', 'extra':'variant_means',
        'side':['Strict field counts:','C01: 1/48','C05: 2/48','C15: 1/48.','4 of 144 DG units.'],
        'insert_caption':['Broad or multifield maps can still be spatial.','Counts use ≥80% dominant-component mass.'],
        'insert_protocol':'DG 16 · three seeds · archived frozen maps · criterion-dependent',
        'survey_subtitle':'3b  Average the three seeds of each variant',
        'survey_caption':['Three-seed averages retain the pooled pattern.','55 variants; one mixed-age variant is omitted.','Associations still mix architecture families.'],
        'claim':'The coverage advantage does not coincide with more units passing the strict single-field criterion.',
        'takehome':'Broader exploration does not establish compact fields or goal control.',
        'question':'What field compactness is necessary for a useful internal goal?',
        'risk':'Very few units pass a stringent classification rule; the counts do not imply an '
               'absence of spatial coding or equivalence between designs. Averaging seeds repeats '
               'the same survey, rather than providing independent replication.',
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


def marker_labels(layer, names, x, y, colors=None):
    for i,name in enumerate(names):
        text(layer,f'{x}-{y}-marker-{i}',x,y+i*18,[name],30,color=colors[i] if colors else '#111111')


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
        for (bin_x,bin_y),shared in labels.groupby(['peak_x_bin','peak_y_bin']):
            point=shared.iloc[0]
            xx = 190+(point.peak_x_bin+.5)/19*w
            yy = map_y+h-(point.peak_y_bin+.5)/19*h
            # Four selected IDs leave all sixteen marker positions visible.
            message=','.join(map(str,sorted(shared.unit.astype(int))))
            label_x=min(max(xx+2,194),190+w-(len(message)*6.7+2))
            label_y=min(max(yy-2,map_y+10),map_y+h-4)
            text(layer,f'{condition}-peak-id-{int(point.unit)}',label_x,label_y,[message],30)
            el=find(layer,f'{condition}-peak-id-{int(point.unit)}')
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
        ax.set(title=f'{architecture} Reward\n/ step × 10³',ylim=(0,.6),yticks=[0,.3,.6])
        g.record('transfer_compact',architecture,'0–75M mean reward',group,'full_0_75m',1000,
                 protocol='joint-system transfer, three downstream seeds',dg_units=64)
    g.finish(fig,Candidate('transfer_compact','Joint-system transfer',(191.5,76.2),'Version A','','','',''))
    import_existing(g,'predictor',OLD_CANDIDATES/'05_predictor_loops_and_coverage.svg',(191.5,76.2))
    path=ROOT/'06_experiments/results/A0_poster_analysis_20260926/c15_variants/trajectory_per_run.csv'
    g.sources[str(path.relative_to(ROOT))]=sha(path)
    mono=g.read(FLAT/'mono_field_peak_counts.csv')
    selected=mono[mono.condition.isin(('C01','C05','C15'))]
    if len(selected)!=9 or selected.mono_units.sum()!=4 or not (selected.dg_units==16).all():
        raise ValueError('Strict-field comparison changed')
    method=FLAT/'mono_field_method.json';g.sources[str(method.relative_to(ROOT))]=sha(method)
    if json.loads(method.read_text())['min_dominant_component_mass_at_each_threshold']!=.8:
        raise ValueError('Strict field criterion changed')
    fig,ax=plt.subplots(figsize=(191.5/25.4,76.2/25.4),layout='constrained')
    dots(ax,selected,'mono_units',('C01','C05','C15'))
    ax.set(title='Strict single-field units / 16',ylim=(-.05,2),yticks=[0,1,2])
    g.record('compact_fields','A','strict field count',selected,'mono_units',dg_units=16,
             protocol='archived frozen maps, dominant mass ≥80%')
    g.finish(fig,Candidate('compact_fields','Compact fields',(191.5,76.2),'Version C','','','',''))


def scatter_panel(ax, data, x, title, x_label, *, marker_size=65):
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
    means=[];excluded=[]
    for (condition,family),group in data.groupby(['condition_original','family']):
        if len(group)!=3 or group.seed.nunique()!=3:
            raise ValueError('Variant means require three distinct seeds')
        if group.frames.nunique()!=1:
            excluded.append({'condition':condition,'reason':'checkpoint age differs across seeds',
                             'seed_ages':group[['seed','frames']].to_dict('records')})
            continue
        means.append({'run_name':condition,'family':family,'condition':family,'seed':np.nan,
                      'variant':condition,'spatial_information':group.spatial_information.mean(),
                      'unique_peak_bins':group.unique_peak_bins.mean(),
                      'prospective_success':group.prospective_success.mean(),
                      'frames':group.frames.iloc[0], 'source_seeds':','.join(map(str,sorted(group.seed.astype(int)))),
                      'study_sha256':group.study_sha256.iloc[0]})
    mean_data=pd.DataFrame(means);mean_data.to_csv(g.out/'variant_means.csv',index=False)
    (g.out/'variant_mean_exclusions.json').write_text(json.dumps(excluded,indent=2)+'\n')
    for key in ('within_families','graph_grouping','variant_means'):
        fig,axes=plt.subplots(1,2,figsize=(383/25.4,112/25.4),sharey=True)
        fig.subplots_adjust(left=.12,right=.965,top=.76,bottom=.235,wspace=.32)
        if key=='within_families':
            specs=[(data[data.family==f],'spatial_information',FAMILY_NAMES[f],'Spatial score') for f in ('CPD','DGP')]
        elif key=='graph_grouping':
            if data.graph_reachability.isna().any():raise ValueError('Missing graph measurements in fixed survey')
            specs=[(data,'graph_reachability','Pooled','Reachable pairs (%)'),
                   (data[data.family=='CPD'],'graph_reachability','CA3 feedback','Reachable pairs (%)')]
        else:
            specs=[(mean_data,'spatial_information','Spatial score','Spatial score'),
                   (mean_data,'unique_peak_bins','Peak diversity','Mean peak bins')]
        for ax,(group,x,name,label) in zip(axes,specs):
            rho=scatter_panel(ax,group,x,'',label)
            displayed_rho=0. if abs(rho)<.005 else rho
            ax.set_title(f'{name}\nn={len(group)}, ρ={displayed_rho:.2f}')
            statistics.append({'panel':key+'_'+name,'x':x,'sample_unit':'variant' if key=='variant_means' else 'run',
                               'n':len(group),'rho':rho,'minimum_frames':int(group.frames.min()),
                               'maximum_frames':int(group.frames.max())})
            g.record(key,name,'recorded DG target-hit fraction',group,'prospective_success',100,
                     dg_units=16,protocol='three-seed variant mean' if key=='variant_means' else 'fixed-DG-16 run survey')
        axes[0].set_ylabel('DG hits (%)')
        g.finish(fig,Candidate(key,key.replace('_',' '),(383,112),'Additional section 3','','','',''))
    (g.out/'cross_run_statistics.json').write_text(json.dumps(statistics,indent=2)+'\n')
    return data


def place_original(parent, group, x, y):
    copied=copy.deepcopy(group)
    s,_,_=matrix(copied)
    copied.set('transform',f'matrix({s},0,0,{s},{x},{y})')
    parent.append(copied)


def compose_right(root, source_right, g: Gallery, variant_key):
    item=VARIANTS[variant_key];x=442.4
    remove(root,find(root,'poster-right-20260929'))
    layer=ET.SubElement(root,f'{{{SVG}}}g',{'id':'revised-right-'+variant_key,
        f'{{{INK}}}groupmode':'layer',f'{{{INK}}}label':'Results · '+item['title']})
    section(layer,2,'Exploration and its limits',x,220)
    text(layer,'subsection-2a',x,244,['2a  Exploration and revisits'],36,bold=True,color='#253B57')
    text(layer,'design-key',x,263,['C01 no goals · C05 uniform goals · C15 frontier goals'],30)
    place_original(layer,find(source_right,'right-exploration-plot'),x,276)
    text(layer,'coverage-gain',x,373,['C15 / C01: 2.1× coverage AUC; 2.7× cells.'],40,bold=True)
    text(layer,'return-definition',x,397,['Mobile path >500 units; return displacement <100.'],40)
    text(layer,'subsection-2b',x,421,['2b  Goal-specific control remains weak'],36,bold=True,color='#253B57')
    place_original(layer,find(source_right,'right-control-plot'),x,431)
    text(layer,'goal-control-caption',645.4,442,
         ['C15 lift <1 in all seeds.','Goal changes barely','alter raw logits.','DG hits do not verify','physical arrival.'],40)
    text(layer,'control-reference',x,525,['Shuffled target reference = 1.'],30)
    text(layer,'subsection-2c',x,550,[item['insert_title']],36,bold=True,color='#253B57')
    embed(layer,g.out/f"{item['candidate']}.svg",'new-result-'+variant_key,x,560,191.5)
    text(layer,'new-result-components',645.4,570,item['side'],40)
    text(layer,'new-result-caption',x,657,item['insert_caption'],40)
    text(layer,'new-result-protocol',x,695,[item['insert_protocol']],30)

    section(layer,3,'Cross-run architecture survey',x,725)
    text(layer,'survey-protocol',x,746,['3a  DG 16 · 168 runs / 56 variants · 25–150M'],36,bold=True,color='#253B57')
    embed(layer,g.out/'survey_main.svg','survey-main',x,755,383)
    text(layer,'subsection-3b',x,908,[item['survey_subtitle']],36,bold=True,color='#253B57')
    embed(layer,g.out/f"{item['extra']}.svg",'survey-extra-'+variant_key,x,921,383)
    text(layer,'survey-interpretation',x,1054,item['survey_caption'],40)
    ET.SubElement(layer,f'{{{SVG}}}rect',{'id':'takehome-box','x':str(x),'y':'1100',
        'width':'383','height':'54','rx':'2','fill':'#F0F3F6'})
    text(layer,'takehome-label',x+8,1113,['Take-home'],30,bold=True,color='#253B57')
    paragraph(layer,'takehome',x+8,1130,item['takehome'],367,40,True)
    text(layer,'reference-title',x,1165,['References'],30,bold=True)
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    plots=args.out/'plots';plots.mkdir(exist_ok=True)
    font=style();g=Gallery(plots)
    digest=sha(args.source)
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
    data=survey(g);inserted_results(g)
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
    report={'schema':'intrmotiv/poster-layout-alternatives/v1','source':str(args.source),'source_sha256':digest,
            'font_path':font,'plot_font_pt':30,'body_font_pt':40,'section_font_pt':48,'subsection_font_pt':36,
            'sources':g.sources,'variants':VARIANTS,'left_data_preservation':left_audit,'quality':results,
            'study_metadata':'Original study SHA-256 values retained in point tables; no study or telemetry changes.'}
    (args.out/'manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    if sha(args.source)!=digest:raise ValueError('Supplied SVG changed')
    print(f'Three A0 alternatives written to {args.out}')


if __name__=='__main__':
    main()
