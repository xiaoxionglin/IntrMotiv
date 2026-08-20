# Agent Notes

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
