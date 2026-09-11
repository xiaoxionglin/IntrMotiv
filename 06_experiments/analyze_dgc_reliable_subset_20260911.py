"""Screen small reliable target sets from canonical edge/unit CSVs.

Rates are observational prospective counts, conditional on previously known
edges. Cutoffs (20 attempts, 80%/90%) are transparent reporting thresholds,
not training changes or proof from matched command interventions.
"""
import argparse
import csv
import math
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('spatial_dir',type=Path); p.add_argument('output_dir',type=Path)
    args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((args.spatial_dir/'per_snapshot.csv').open()))
    identities={(r['run_name'],int(r['target_env_steps'])):r for r in rows}
    units={(r['run_name'],int(r['target_env_steps']),int(r['unit_id'])):r
           for r in csv.DictReader((args.spatial_dir/'per_unit.csv').open())}
    grouped={key:[] for key in identities}
    for row in csv.DictReader((args.spatial_dir/'graph_edge.csv').open()):
        key=(row['run_name'],int(row['target_env_steps']))
        if key not in grouped: raise ValueError(key)
        if int(row['reliable']) and float(row['prospective_attempts'])>=20:
            row['observed_success_rate']=float(row['prospective_successes'])/float(row['prospective_attempts'])
            grouped[key].append(row)
    summaries=[]; edges=[]
    for key,values in grouped.items():
        identity=identities[key]
        result={name:identity[name] for name in ['run_name','base','capacity','seed','target_env_steps']}
        result['reliable_pairs_with_20_prospective_attempts']=len(values)
        for rate,label in [(0.8,'80'),(0.9,'90')]:
            selected=[r for r in values if r['observed_success_rate']>=rate]
            targets={int(r['target_unit']) for r in selected}
            qualified={t for t in targets if int(units[(*key,t)]['mono_field'])}
            peak_bins={(float(units[(*key,t)]['dominant_peak_x']),float(units[(*key,t)]['dominant_peak_y']))
                       for t in targets if math.isfinite(float(units[(*key,t)]['dominant_peak_x']))}
            result[f'pairs_at_{label}']=len(selected)
            result[f'targets_at_{label}']=len(targets)
            result[f'distinct_target_peaks_at_{label}']=len(peak_bins)
            result[f'mono_targets_at_{label}']=len(qualified)
            result[f'pairs_also_current_posterior_{label}']=sum(float(r['posterior_reliability'])>=rate for r in selected)
        for row in sorted(values,key=lambda r:(-r['observed_success_rate'],-float(r['prospective_attempts'])))[:12]:
            source=int(row['source_unit']); target=int(row['target_unit'])
            edges.append({**{name:identity[name] for name in ['run_name','base','capacity','seed','target_env_steps']},
                **{name:row[name] for name in ['source_unit','target_unit','prospective_attempts','prospective_successes',
                    'observed_success_rate','posterior_reliability','attempts','tctrl','endpoint_peak_distance']},
                'source_mono_field':units[(*key,source)]['mono_field'],
                'target_mono_field':units[(*key,target)]['mono_field'],
                'target_peak_x':units[(*key,target)]['dominant_peak_x'],
                'target_peak_y':units[(*key,target)]['dominant_peak_y']})
        summaries.append(result)
    for filename,data in [('reliable_subset_summary.csv',summaries),('top_tested_edges.csv',edges)]:
        if not data: raise ValueError('no rows')
        with (args.output_dir/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)


if __name__=='__main__': main()
