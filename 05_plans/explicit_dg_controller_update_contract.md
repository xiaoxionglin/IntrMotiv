# Explicit DG–Controller Update Contract

Status: implemented and submitted on 2026-09-05.

## Contract

The active IntrMotiv learner retains the original push–pull objectives:

\[
r_{\mathrm{enc}}=\beta d, \qquad r_{\mathrm{dec}}=\beta(E-d).
\]

Each minibatch has one differentiable DG/CA3 forward. The controller receives a
view with only the CA3 prefix detached; visual bypass, target, geometry, and
manager-mode features retain their controller gradients. Encoder credit is
intersected with the recomputed active DG row. Simultaneous training performs
one backward on the summed loss, while iterative encoder and decoder phases
backpropagate only their selected loss. No gradient deletion, gradient flip, or
second encoder forward is used by the active learner.

With `dg_batchnorm_semantics=running_consistent`, actors and learners normalize
DG logits with stored running statistics. Only simultaneous or encoder-active
learner forwards update them, once per minibatch. The first eligible forward
initializes the statistics from its batch.

Landmark replacement is atomic under the policy lock. It resets the DG row and
optimizer state, invalidates representation-dependent state, zeros the matching
target-ID FiLM row and optimizer state, and advances the representation
generation. Samples from an older stored option generation are excluded from
both PPO and encoder losses.

## Diagnostic batch

- Study: `encoder_decoder_update_contract_preflight_20260905`
- Batch: `intrmotiv_encoder_decoder_update_contract_preflight_20260905`
- Project: `SF_IntrMotiv_EncoderDecoderUpdateContractPreflight`
- Matrix: C15-FiLM × `{ARR,SRC}` × `{MON,DIRS,DIRO,PREDS,PREDO}` × seed 99
- Training horizon: 75M environment steps per run
- Schema/workflow: `intrmotiv/study/v1`, `1.4.1`
- Validated StudySpec SHA-256: `1cc2a8debf8cfc49a9dc5d17b0b270180f3b3aaa47edbde08e536b10363cc291`
- StudySpec: `hpc_runs/studies/encoder_decoder_update_contract_preflight.study.json`
- Adapter: `hpc_runs/encoder_decoder_update_contract_preflight.py`
- Runtime gate: `hpc_runs/analyze_encoder_decoder_update_contract.py`

Early synchronized analyses are declared at 5M, 25M, 50M, and 75M. The batch
does not trigger a dependent production submission. It is scientifically
separate from the already-running source-credit workflow-1.4.1 batch.

## NEMO2 artifacts

- Validated study and rendered commands:
  `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/encoder_decoder_update_contract_preflight_20260905/`
- Print-only submission:
  `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_encoder_decoder_update_contract_preflight_20260905/20260905_print_only/`
- Submitted jobs and logs:
  `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/_slurm/intrmotiv_encoder_decoder_update_contract_preflight_20260905/20260905_submitted/`
