#!/usr/bin/env python3
"""Render original C15 from completed canonical telemetry; run on NEMO2.

Usage: python render_c15_completed_place_fields.py TELEMETRY_ROOT OUTPUT_DIR
Run from SF_hipposlam with PYTHONPATH=. Raw artifacts never leave the workspace.
No rollout is performed. Maps are divided by each unit's own maximum absolute
value for shape inspection; titles give that scale. No spatial smoothing.
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, colors
from sf_working_directories.IntrMotiv.evaluation.analyze_place_field_manifest import (
    artifact_for_suffix, derive_row, per_unit_rows,
)

root, out = map(Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=True)
font = Path(font_manager.findfont('DejaVu Sans', fallback_to_default=False))
assert font.is_file() and font.suffix.lower() in ('.ttf', '.otf')
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':28, 'axes.titlesize':30,
                     'axes.labelsize':28, 'xtick.labelsize':26, 'ytick.labelsize':26,
                     'pdf.fonttype':42, 'svg.fonttype':'none'})
manifest = root/'analysis_manifest.tsv'
rows = pd.read_csv(manifest, sep='\t', dtype=str)
rows = rows[rows.condition.eq('c15_topology_ucb_direct_o1')]
assert len(rows)==7
metrics, units, provenance = [], [], []
cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('#d1d5db')
div = plt.get_cmap('RdBu_r').copy(); div.set_bad('#d1d5db')

def save(fig, name):
    fig.savefig(out/f'{name}.png', dpi=120, facecolor='white')
    fig.savefig(out/f'{name}.svg', facecolor='white')
    plt.close(fig)

def spatial(ax):
    ax.set_xticks([0,9,18]); ax.set_yticks([0,9,18])
    ax.set_xlim(-.5,18.5); ax.set_ylim(-.5,18.5)

for item in rows.to_dict('records'):
    path = artifact_for_suffix(root/'raw', item['label_suffix'])
    data = np.load(path, allow_pickle=False)
    occ = data['occupancy']; valid = occ>0
    assert occ.shape==(19,19) and data['rate_maps'].shape==(19,19,16)
    record = derive_row(item,path)
    unit_rows = per_unit_rows(item,path); units.extend(unit_rows)
    record['multi_region_units_raw_half_peak'] = sum(u['connected_components_half_peak']>1 for u in unit_rows)
    record['occupancy_top10_cell_fraction'] = float(np.sort(occ.ravel())[-10:].sum()/occ.sum())
    record['occupancy_median_visited'] = float(np.median(occ[valid]))
    peaks = np.array([np.unravel_index(np.nanargmax(data['rate_maps'][:,:,u]),occ.shape) for u in range(16)])
    distances = np.linalg.norm(peaks[:,None]-peaks[None,:],axis=-1); np.fill_diagonal(distances,np.inf)
    record['peak_nearest_neighbor_median_bins'] = float(np.median(distances.min(axis=1)))
    record['peak_nearest_neighbor_min_bins'] = float(distances.min())
    metrics.append(record)
    provenance.append({'label_suffix':item['label_suffix'],'artifact':str(path),
                       'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                       'checkpoint_frames':int(item['checkpoint_frames'])})
    if int(item['target_frames']) != 100000000: continue
    seed = item['seed']
    for key, tag, palette in [('rate_maps','rate',cmap),('pre_threshold_rate_maps','prethreshold',div)]:
        maps = data[key]
        # Normalize each unit explicitly; retain its numeric scale in the title.
        norm = colors.Normalize(-1 if tag=='prethreshold' else 0, 1)
        for start in (0,4,8,12):
            fig, axs = plt.subplots(2,2,figsize=(10,11),layout='constrained')
            for ax, unit in zip(axs.flat,range(start,start+4)):
                scale = float(np.nanmax(np.abs(maps[:,:,unit][valid])))
                assert scale > 0
                im = ax.imshow(np.ma.masked_where(~valid,maps[:,:,unit]/scale).T,
                               origin='lower', interpolation='nearest', cmap=palette,norm=norm)
                spatial(ax); ax.set_title(f'DG {unit:02d}\nscale {scale:.2f}', fontsize=26)
            fig.suptitle(f'C15 · seed {seed} · '+('DG activity' if tag=='rate' else 'DG logits'),fontsize=30)
            fig.supxlabel('x bin (100 position units/bin)')
            fig.supylabel('y bin')
            cb=fig.colorbar(im,ax=axs,orientation='horizontal',fraction=.06,pad=.02,shrink=.9)
            cb.set_ticks([-1,0,1] if tag=='prethreshold' else [0,.5,1])
            cb.ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
            cb.set_label('Mean logit / unit scale' if tag=='prethreshold' else 'Mean activity / unit peak')
            save(fig,f'c15_s{seed}_{tag}_{start:02d}')
    pose = pd.read_csv(path.with_name('pose.csv'))
    fig, axs = plt.subplots(1,2,figsize=(12,6.5),layout='constrained')
    im=axs[0].imshow(np.ma.masked_where(~valid,occ).T,origin='lower',cmap=cmap,
                    norm=colors.LogNorm(1,occ.max()),interpolation='nearest')
    axs[0].set_title(f'Occupancy: {valid.sum()}/361')
    fig.colorbar(im,ax=axs[0],shrink=.7).set_label('Samples (log scale)')
    axs[1].imshow(np.where(valid,1,np.nan).T,origin='lower',cmap=colors.ListedColormap(['#f1f5f9']),interpolation='nearest')
    axs[1].set_facecolor('#d1d5db')
    axs[1].scatter(peaks[:,0],peaks[:,1],s=100,color='#0072B2')
    axs[1].set_title('DG peaks (16)')
    for ax in axs: spatial(ax); ax.set_xlabel('x bin'); ax.set_ylabel('y bin')
    fig.suptitle(f'C15 · seed {seed} · terminal probe',fontsize=30)
    save(fig,f'c15_s{seed}_context')
    # Four fixed windows span the probe; break paths at recorded episode boundaries.
    fig, axs=plt.subplots(2,2,figsize=(10,11),layout='constrained')
    for ax,start in zip(axs.flat,np.linspace(0,len(pose)-300,4,dtype=int)):
        chunk=pose.iloc[start:start+300]
        for _,segment in chunk.groupby('num_traj',sort=False):
            ax.plot(segment.x/100-1.5,segment.y/100-1.5,lw=1.4,color='#0072B2')
        ax.scatter(chunk.iloc[0].x/100-1.5,chunk.iloc[0].y/100-1.5,c='#E69F00',s=90,zorder=4)
        spatial(ax); ax.set_title(f'{start}–{start+len(chunk)-1}')
    fig.suptitle(f'C15 · seed {seed} · path windows',fontsize=30)
    fig.supxlabel('x bin · orange = window start'); fig.supylabel('y bin')
    save(fig,f'c15_s{seed}_paths')
pd.DataFrame(metrics).to_csv(out/'c15_metrics.csv',index=False)
pd.DataFrame(units).to_csv(out/'c15_per_unit.csv',index=False)
(out/'provenance.json').write_text(json.dumps({'source_manifest':str(manifest),
    'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
    'protocol':'original 10000-decision stochastic probes; historical manifest; no new StudySpec',
    'pose_alignment':'legacy artifacts, no pose_alignment metadata; possible one-decision offset',
    'display_normalization':'each map divided by its own max absolute visited-cell value; title gives denominator',
    'map_axes':'original array axes x,y; transposed for Cartesian display',
    'font':str(font),'matplotlib':matplotlib.__version__,'numpy':np.__version__,
    'artifacts':provenance},indent=2)+'\n')
print(pd.DataFrame(metrics)[['seed','checkpoint_frames','visited_cells','mono_field_fraction','multi_region_units_raw_half_peak','occupancy_top10_cell_fraction']].to_string(index=False))
