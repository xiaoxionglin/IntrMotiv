# PPO-budget replay extension — 13 September 2026

User authorized adding runs with a replay ratio matching PPO's one fresh pass.
The existing 18-run study is preserved; this supplementary 12-run StudySpec uses
both architectures and seeds 8, 99, 123. Existing PPO runs serve as references.

Study: `hpc_runs/studies/full_system_controller_rr1.study.json`.
Schema `intrmotiv/study/v1`, workflow `1.8.1`, SHA
`74aa87acdcc44a6085a4c8f952a893527cc61bf4b9d1346ab3b482647fe44e07`.

Every 2,048 accepted decisions after the original 16,384-decision warmup:

| Arm | Main examples | Additional auxiliary examples | Total ratio |
|---|---:|---:|---:|
| DDQN | 2,048 | 0 | 1 |
| DDQN+HER | 1,024 | Up to 1,024 | At most 1 |

HER eligibility failures are not backfilled. Each objective retains its separate
mean loss, auxiliary coefficient 1.0. Consequently HER does not have equal main
training exposure to plain DDQN in this extension. This is an efficiency
comparison changing both replay ratio and batch size, not an isolated ratio
ablation. The existing learner needs no code modification.

Target refresh is every three completed updates, nominally 6,144 accepted
post-warmup decisions versus the parent's 6,400. Learning rate remains 0.0002;
no assumption that larger batches require a scaled learning rate. Fresh DG,
graph cadence, stored memory, STOP routing, epsilon schedule and terminal
semantics are preserved. Fresh 300M horizons, 128GiB/16CPU/L40S/48h allocations,
and checkpoint/telemetry schedules are inherited. Learners use GPU; software
simulation and replay preparation use CPU.

Immutable source clone:
`/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_controller_rr1_20260913`.
Only new study and adapter files are added to the qualified released runtime.
39 focused schedule/release/workflow tests passed remotely. Canonical print-only
review and workspace-path audit passed for all twelve commands. The larger
batches have not yet had a full environment throughput qualification; observed
throughput and memory must be reported as measurements, not promised gains.

Output batch: `intrmotiv_full_system_controller_rr1_20260913` under the allocated
training workspace. W&B group is shared with the parent production study;
new run names begin `FSCR1_`. Separate canonical manifest prevents mutating the
parent's declared 18-row matrix. Monitoring remains paused at user request.

Reusable lesson: distinguish replay examples per decision, optimizer batch size,
and optimizer calls per second. Equal total HER budgets require explicitly
recording reduced main exposure and the eligibility-dependent actual ratio.
Reuse parameterized learner settings and the canonical submission workflow.
