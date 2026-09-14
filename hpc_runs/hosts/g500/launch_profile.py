"""Dispatch an authorized G500 profile; pass W&B credentials only over SSH stdin."""
import argparse
import json
import netrc
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--gpus", type=int, nargs="+", default=[0])
    p.add_argument("--seconds", type=int, default=900)
    p.add_argument("--frames", type=int, default=131072)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if not a.name.replace("_", "").isalnum():
        p.error("name must contain only letters, digits, underscores")
    root = "/scratch/lin/IntrMotiv"
    source = root + "/src/SF_hipposlam_dg_neighborhood_20260914"
    args = [root + "/envs/SF_git/bin/python", root + "/tools/g500/profile_training.py",
            root + "/tools/g500/profile_parent.study.json", "DGP_C15_HIT_STOP_FILM_S99",
            root + "/train_dir/resource_profile_20260914/" + a.name,
            f"--workers={a.workers}", f"--batch-size={a.batch_size}", f"--seconds={a.seconds}", f"--frames={a.frames}",
            "--gpus", *map(str, a.gpus)]
    if a.execute:
        args.append("--execute")
    code = "import os,sys,subprocess\n"
    if a.execute:
        code += "os.environ['WANDB_API_KEY']=sys.stdin.readline().strip()\n"
        code += f"log=open({root + '/logs/profile-' + a.name + '.log'!r},'x')\n"
        code += f"proc=subprocess.Popen({args!r},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint('profiler_pid',proc.pid)\n"
    else:
        code += f"raise SystemExit(subprocess.call({args!r}))\n"
    command = f"source {root}/tools/g500/activate.sh; export INTRMOTIV_SOURCE={source} PYTHONPATH={source}; python -c " + shlex.quote(code)
    key = netrc.netrc().authenticators("api.wandb.ai")[2] + "\n" if a.execute else ""
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
                             "lin@g500-2603n3.bcf.privat", command], input=key, text=True, capture_output=True)
    print(result.stdout)
    print(result.stderr)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
