# Agent Notes

## IntrMotiv Architecture Intent

- `layer2_resnet18` is intentionally an ImageNet-pretrained ResNet-18 trunk
  through layer 2, kept fixed during IntrMotiv training. The DG projection and
  its BatchNorm running statistics are the trainable visual-landmark layer.

## NEMO2 Access

- Do not blindly retry SSH to NEMO2 after authentication failures.
- If SSH reports `Permission denied`, `ssh_askpass`, `Too many authentication failures`, or appears to require keyboard-interactive auth, assume the user may need to manually log in with OTP first.
- Ask the user to complete a manual login from their terminal, for example `ssh nemo2`, before retrying automated SSH checks.
- When retrying after user confirmation, prefer an explicit key/config-minimal command if needed, e.g. `ssh -F /dev/null -o IdentityAgent=none -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519_NEMO2_v2 fr_xl1014@nemo2-login.nemo.uni-freiburg.de`.

## NEMO2 Storage Policy

- Store all training data in an allocated NEMO2 workspace, never under the home filesystem.
- Use `/work/classic/fr_xl1014-train` as the workspace root for this project.
- This includes `train_dir`, model checkpoints, milestone and best checkpoints, TensorBoard events, W&B run data, Slurm stdout/stderr, environment caches, rollout data, and generated datasets.
- Keep the home folder strictly limited to source code and lightweight analysis scripts, reports, and plots.
- Before submitting a run, verify that every training-output, logging, cache, and temporary-data path resolves into the intended workspace.
- Do not launch training if any bulk output path points into `/home/fr/fr_xl1014` or a source checkout located there. Use lightweight source-side symlinks or configuration pointers to workspace data only when compatibility requires them.

## Reusable Place-Field Telemetry

- Before adding DG telemetry or place-field analysis, read
  `04_implementation/reusable_place_field_telemetry.md` in this vault and reuse
  the established manifest-driven evaluation package.
- The authoritative NEMO2 implementation is under
  `/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam/sf_working_directories/IntrMotiv/evaluation/`.
  In particular, use `place_fields.py`, `submit_place_field_sweep.py`,
  `run_place_field_sweep_single.sh`, `summarize_place_fields.py`,
  `analyze_place_field_manifest.py`, `plot_place_field_trajectories.py`, and
  `map_stability.py` rather than creating batch-specific `enjoy` scripts.
- On NEMO2, submit each telemetry manifest row as an ordinary independent
  `sbatch` job through `submit_place_field_sweep.py`; do not use a Slurm array
  for new telemetry. Always run its print-only review before `--submit`.
  `run_place_field_sweep_array.sh` is compatibility-only for historical or
  already queued arrays. This policy is intentional because ordinary jobs are
  evaluated more reliably by the current NEMO2 scheduler.
- The standard comparison is a 10k-decision rollout at five checkpoints for
  seed 99 plus terminal checkpoints for seeds 8 and 123. Use a Slurm preflight;
  do not run DMLab telemetry on the login node.
- Keep raw NPZs, full plot sets, Slurm logs, caches, and temporary files in
  `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/`.
  Copy only lightweight summaries and selected figures into this vault.
- Report active-only map cosine and peak diversity together with silent units,
  spatial information, and pre-threshold maps. Never infer place-field
  diversity from online DG density alone, and do not treat policy-driven
  checkpoint rollouts as a fixed-trajectory drift test.
- Preserve the manifest and NPZ contracts when extending the evaluator. Add
  metrics and arrays compatibly, update focused tests and the reusable
  telemetry document, and link each completed analysis from the vault index.
