"""Dispatch an already reviewed direct-queue manifest to G500.

This is intended for explicit recovery subsets that cannot be represented as a
Cartesian StudySpec without rerunning healthy cells. The manifest is accepted
only when its canonical SHA-256 matches the caller-provided review token.
"""
import argparse
import netrc
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="Absolute manifest path on G500")
    parser.add_argument("--execute", required=True, help="Reviewed manifest SHA-256")
    args = parser.parse_args()
    root = "/scratch/lin/IntrMotiv"
    runner_code = f'''import hashlib,json
from pathlib import Path
from hpc_runs.intrmotiv_study.direct import run_queue
from hpc_runs.hosts.g500.profile_training import resources
path=Path({args.manifest!r})
manifest=json.loads(path.read_text())
claimed=manifest["manifest_sha256"]
payload=dict(manifest)
payload.pop("manifest_sha256")
actual=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
if claimed != {args.execute!r} or actual != claimed:
    raise RuntimeError("Recovery manifest differs from print-only review")
run_queue(manifest,resources)
'''
    log = f"{root}/logs/direct-recovery-{args.execute[:16]}.log"
    bootstrap_code = f'''import os,subprocess,sys
os.environ["WANDB_API_KEY"]=sys.stdin.readline().strip()
handle=open({log!r},"a")
process=subprocess.Popen(
    [{root + '/envs/SF_git/bin/python'!r},"-c",{runner_code!r}],
    stdin=subprocess.DEVNULL,
    stdout=handle,
    stderr=subprocess.STDOUT,
    start_new_session=True,
    env=os.environ,
)
print(process.pid)
'''
    remote = (
        f"source {root}/tools/g500/activate.sh; "
        f"export PYTHONPATH={shlex.quote('/scratch/lin/IntrMotiv/src/SF_hipposlam_dg_neighborhood_gpu_core_stale_skip_20260915')}; "
        f"{root}/envs/SF_git/bin/python -c {shlex.quote(bootstrap_code)}"
    )
    credential = netrc.netrc().authenticators("api.wandb.ai")[2] + "\n"
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", "lin@g500-2603n3.bcf.privat", remote],
        input=credential,
        text=True,
        capture_output=True,
    )
    print(result.stdout, end="")
    print(result.stderr, end="")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
