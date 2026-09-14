# G500 IntrMotiv environment

Setup started 2026-09-14. Status: Python installed; CUDA dependencies installing;
private source and DMLab transfer awaits explicit user approval.

## Connection and storage

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 lin@g500-2603n3.bcf.privat
```

SSH succeeded from the desktop with network-enabled execution. Sandboxed SSH
reported a DNS failure; this did not indicate a server or authentication failure.
The account is `lin`, home `/homeatlas/lin`, group `leibold`.

Use `/scratch/lin/IntrMotiv` for this host's environment, training outputs,
caches, temporary files, logs, and setup artifacts. This is local NVMe storage
with approximately 1.3 TB available at inspection. Scratch retention and quotas
have not been established. The NEMO2 workspace path does not apply to this host.

Observed hardware: Ubuntu 24.04.4, 192 logical CPUs, 377 GiB RAM, two NVIDIA
RTX PRO 6000 Blackwell GPUs with 97,887 MiB each, driver 580.159.03. Both GPUs
have existing compute processes. No `sbatch` was found on the default PATH.

## Environment

- Python: `/scratch/lin/IntrMotiv/envs/SF_git/bin/python` (3.10.21).
- This is a virtual environment, not a conda environment.
- Bootstrap: system Python venv plus uv 0.12.13, with managed Python under
  `/scratch/lin/IntrMotiv/tools/python`.
- Selected GPU packages: torch 2.8.0 and torchvision 0.23.0, CUDA 12.8 wheels,
  from the [official matched-version instructions](https://pytorch.org/get-started/previous-versions/).
- Installation log: `/scratch/lin/IntrMotiv/logs/install-torch.log`.

## Pending runtime deployment

The clean local runtime checkout is `/home/xiaoxiong/SFgit/SF_hipposlam`, commit
`d94155f9be0436828ee9a744b57097db07022344`. A tracked-source archive is staged
locally at `/tmp/intrmotiv-g500-source.tar`. Do not interpret this checkout as
the newest qualified NEMO2 experiment branch without checking study provenance.

The existing local DMLab wheel is
`/home/xiaoxiong/deepmind_lab-1.0-py3-none-any.whl`; its recorded SHA-256 is
`a5e2fd32773193bc643acc9d9dd2006feeb0917ab80e4dae462fd60c44148df1`.
Custom levels require the runtime repository's `deepmindlab_patch/` assets.
The patch script currently requires `CONDA_PREFIX` even though it resolves
site-packages using the active Python; handle this explicitly for the venv.

The checkout's `requirements.txt` contains a different user's absolute wheel
path and a Git editable dependency; do not install it verbatim. Reuse package
metadata in `setup.py`, pin the selected torch/torchvision pair, NumPy below 2,
and setuptools below 81 for DMLab's `pkg_resources` import.

Before declaring readiness, install the source and patched DMLab, run `pip check`,
verify GPU tensor operations and pretrained ResNet inference, reset and step
the custom no-reward fixed-length level, and run focused runtime/workflow tests.
Training has not been launched. Use the
[standardized workflow](standardized_study_workflow.md) for subsequent studies;
existing NEMO2 Slurm launch commands are not directly applicable here.

## Reusable lessons

Inspect the target first, use its writable scratch space, and install a fresh
matched GPU stack instead of copying a large desktop conda environment. Reuse
the existing DMLab binary and repository patch rather than rebuilding the engine.
An automatic approval review blocked transferring private source and binary
artifacts until the user explicitly approves those payloads for this host.
