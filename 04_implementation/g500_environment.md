# G500 IntrMotiv environment

Setup completed 2026-09-14. Status: environment verified on both GPUs, custom
DMLab reset/step passed, and 68 focused runtime/workflow tests passed. No training
was launched; experiment-specific preflights remain part of each study.

## Use

```bash
ssh lin@g500-2603n3.bcf.privat
source /scratch/lin/IntrMotiv/tools/g500/activate.sh
python /scratch/lin/IntrMotiv/tools/g500/smoke_environment.py
```

Activation selects the Python environment and source import path, directs
caches/logs/temporary data to scratch, and changes into `/scratch/lin/IntrMotiv`
so Sample Factory's default relative `train_dir` also lands there. In launch
commands, prefer explicit `--train_dir="$INTRMOTIV_TRAIN_DIR"` and
`--dmlab_level_cache_path="$INTRMOTIV_ROOT/cache/dmlab"`. GPU selection remains
explicit per run (`CUDA_VISIBLE_DEVICES=0` or `1`); existing jobs share both GPUs.

Maintained activation and smoke helpers: [hpc_runs/hosts/g500](../hpc_runs/hosts/g500/).

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
- DMLab required SDL2, which was missing on the host. Ubuntu package
  `libsdl2-2.0-0` version `2.30.0+dfsg-1ubuntu3.1` was downloaded with
  `apt-get download` and extracted with `dpkg-deb -x` into
  `/scratch/lin/IntrMotiv/tools/sysroot`. Activation sets its library path;
  all engine shared-library dependencies resolve. No system packages changed.
- Resolved packages: [requirements-verified.txt](../06_experiments/data/g500_setup_20260914/requirements-verified.txt).

## Runtime deployment and provenance

The clean local runtime checkout is `/home/xiaoxiong/SFgit/SF_hipposlam`, commit
`d94155f9be0436828ee9a744b57097db07022344`. A tracked-source archive is staged
locally at `/tmp/intrmotiv-g500-source.tar` and extracted remotely to
`/scratch/lin/IntrMotiv/src/SF_hipposlam`. This is a source snapshot without Git
history. Do not interpret this checkout as
the newest qualified NEMO2 experiment branch without checking study provenance.

The existing local DMLab wheel is
`/home/xiaoxiong/deepmind_lab-1.0-py3-none-any.whl`; its recorded SHA-256 is
`a5e2fd32773193bc643acc9d9dd2006feeb0917ab80e4dae462fd60c44148df1`.
Custom levels use the runtime repository's `deepmindlab_patch/` assets.
The patch script currently requires `CONDA_PREFIX` even though it resolves
site-packages using the active Python. Setup ran the unchanged patcher with
`CONDA_PREFIX="$VIRTUAL_ENV"` scoped to that one command.

Workflow 1.8.1 (`intrmotiv/study/v1`) was synchronized from the vault over the
snapshot's 1.7.1 package. The matching existing checkpoint selector was copied
from `/tmp/intrmotiv_stored_release_20260912`: its only differences are adding
`target_frames=TARGET_FRAMES` to `select_checkpoints` and iterating that argument.
Canonical tests and their telemetry-probe study fixture were synchronized too.
This makes no claim that the runtime contains the latest NEMO2 controller or
off-policy experiment branches.

Transferred archive, wheel and ResNet hashes matched the local originals:
[SHA256SUMS](../06_experiments/data/g500_setup_20260914/SHA256SUMS).

The checkout's `requirements.txt` contains a different user's absolute wheel
path and a Git editable dependency; do not install it verbatim. Reuse package
metadata in `setup.py`, pin the selected torch/torchvision pair, NumPy below 2,
and setuptools below 81 for DMLab's `pkg_resources` import.

## Verification

- `python -m pip check`: no broken requirements.
- Both GPUs: CUDA matrix operations and backward pass, followed by pretrained
  ResNet-18 inference; finite results, compute capability 12.0.
- Patched DMLab: software-rendered 96×72 RGB frame, 128 engine frames in
  `openfield_map2_fixed_loc3_fixedlength_noreward`, zero reward and active episode.
- 68 tests passed: core logic repairs, fixed-length level, navigation actions,
  update contract, canonical studies, latest-common collection and telemetry targets.
- Logs: [engine/GPU smoke](../06_experiments/data/g500_setup_20260914/smoke-environment.log)
  and [focused tests](../06_experiments/data/g500_setup_20260914/focused-tests.log).

Use the
[standardized workflow](standardized_study_workflow.md) for subsequent studies;
existing NEMO2 Slurm launch commands are not directly applicable here.

Actual training qualification and resource profiling are tracked in the
[DG neighborhood workstation record](../06_experiments/dg_neighborhood_g500_20260914.md).
That work identified W&B SDK and GPU reward-tensor portability fixes in an
isolated source snapshot. The original setup smoke did not exercise a complete
GPU learner update. Use fixed-frame resource probes and verify real process
completion before choosing worker counts, batch size, or concurrent runs.

## Reusable lessons

Inspect the target first, use its writable scratch space, and install a fresh
matched GPU stack instead of copying a large desktop conda environment. Reuse
the existing DMLab binary and repository patch rather than rebuilding the engine.
Use `ldd` on the DMLab renderer when imports work but engine creation fails;
extracting the missing distribution library into a user-local sysroot resolved
the failure. Package metadata plus `pip check`, actual engine execution and
focused tests were authoritative. The source transfer initially required an
additional approval from automatic review; the user approved it and it completed.
