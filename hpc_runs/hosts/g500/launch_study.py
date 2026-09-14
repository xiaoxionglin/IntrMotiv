"""Print-review or securely dispatch a canonical G500 direct queue."""
import argparse
import netrc
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('study', help='Remote StudySpec path')
    p.add_argument('--source', default='/scratch/lin/IntrMotiv/src/SF_hipposlam_dg_neighborhood_20260914')
    p.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 0, 1])
    p.add_argument('--execute', help='SHA of the reviewed manifest')
    a = p.parse_args()
    root = '/scratch/lin/IntrMotiv'
    command = [root + '/envs/SF_git/bin/python', '-m', 'hpc_runs.intrmotiv_study.direct', a.study,
               '--source', a.source, '--gpus', *map(str, a.gpus)]
    code = 'import os,sys,subprocess\n'
    if a.execute:
        command += ['--execute', a.execute]
        code += "os.environ['WANDB_API_KEY']=sys.stdin.readline().strip()\n"
        code += f"log=open({root + '/logs/direct-' + a.execute[:16] + '.log'!r},'x')\n"
        code += f"process=subprocess.Popen({command!r},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint('queue_pid',process.pid)\n"
    else:
        code += f'raise SystemExit(subprocess.call({command!r}))\n'
    remote = f'source {root}/tools/g500/activate.sh; export PYTHONPATH={shlex.quote(a.source)}; python -c ' + shlex.quote(code)
    credential = netrc.netrc().authenticators('api.wandb.ai')[2]+'\n' if a.execute else ''
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','lin@g500-2603n3.bcf.privat',remote],
                            input=credential,text=True,capture_output=True)
    print(result.stdout)
    print(result.stderr)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
