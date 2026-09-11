"""Thin adapter for the established independent-job telemetry submitter."""
import os
from pathlib import Path
import sys
from .evaluate import main


def run():
    from sf_working_directories.IntrMotiv.evaluation.submit_place_field_sweep import load_manifest
    manifest,index,output=sys.argv[1:]
    row=load_manifest(Path(manifest))[int(index)].values
    if row['family']!='parent':
        raise ValueError('child manifests require a resolved parent linkage')
    sys.argv=[sys.argv[0],'--parent-run-dir',row['run_dir'],'--parent-checkpoint',row['checkpoint'],
              '--output',str(Path(output)/'raw'/row['label_suffix']),
              '--decision-cap',os.environ.get('PLACE_FIELD_MAX_FRAMES','100000')]
    main()


if __name__=='__main__': run()
