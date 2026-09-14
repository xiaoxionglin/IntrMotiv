"""Resource decisions and checkpoint-validation helpers for unattended execution."""
import json
from datetime import datetime, timedelta
import pytest
import torch
from production_neighborhood import concurrency_from_samples
from gate_neighborhood import finite_tensors


@pytest.mark.parametrize('interval,ram,expected',[(60,100,4),(300,100,2),(60,63,2)])
def test_concurrency_uses_progress_and_reserves(tmp_path,interval,ram,expected):
    path=tmp_path/'resources.jsonl'
    path.write_text(json.dumps({'available_ram_gib':ram,'gpus':[{'free_mib':64000}]})+'\n')
    start=datetime(2026,9,14,12)
    for arm in ('BASE','SELF','TEMP','PHYS'):
        lines=[]
        for i in range(3):
            timestamp=(start+timedelta(seconds=i*interval)).strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            lines.append(f'[{timestamp}][1] Total num frames: {(i+1)*32768}.')
        (tmp_path/f'DGN_{arm}_S99.log').write_text('\n'.join(lines))
    slots,evidence=concurrency_from_samples(path)
    assert len(slots)==expected
    assert evidence['aggregate_fps']>0


def test_checkpoint_finiteness_covers_optimizer_nesting():
    assert finite_tensors({'optimizer':{'state':{0:{'exp_avg':torch.zeros(2)}}}})
    assert not finite_tensors({'optimizer':{'state':{0:{'exp_avg':torch.tensor([float('nan')])}}}})
