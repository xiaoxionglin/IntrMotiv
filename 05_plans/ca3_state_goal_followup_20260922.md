# CA3-state goal follow-up — 22 September 2026

Status: follow-up fixes and deferred refinements for the deployed predictive active-goal design.


## 23 September diagnosis: simplify goal recognition back to z-space

Current production diagnostics change the priority:

- `state_shuffle_delta` grows, so the readout state carries useful predictive information;
- `action_shuffle_delta` oscillates around zero, so the innovation predictor is currently using little or no extra information from the executed action sequence;
- raw prediction loss can increase even while state-shuffle separation grows, consistent with a harder or moving target distribution rather than latent collapse;
- `positive_similarity_q10` rapidly saturates at approximately 1 and stays there, making the current action-probe-signature recognition unnecessarily brittle.

The worker already uses the same readout for current state and goal:

$$
z_t=W S_t,
\qquad
z_g=W S_g.
$$

Make this also the canonical goal-recognition space. Replace action-probe-signature success semantics with direct normalized latent similarity:

$$
\boxed{
\operatorname{hit}(S_t,S_g)
=
[\operatorname{has\_DG\_event}(S_t)]
[
\cos(\widehat z_t,\widehat z_g)\ge\tau_z
].
}
$$

where

$$
\widehat z=\frac{z}{\max(\lVert z\rVert,\epsilon)}.
$$

Use the same helper for online graph recognition and contextual HER. DG ID remains only a graph slot/address/bookkeeping field; it does not define semantic success in state-readout mode.

Recalibrate the recognition threshold directly from z-space within-occurrence positive pairs. Keep background/cross-occurrence z-similarity distributions as diagnostics. If positive/background overlap is too large, first add a small trusted-positive z-consistency objective rather than reintroducing a second semantic space.

The previous action-probe signature was introduced because prediction loss alone does not guarantee a meaningful latent metric: an invertible change of coordinates in z can be compensated by the predictor. The current variance/covariance regularization approximately fixes scale/correlation, leaving mostly orthogonal freedom, under which cosine similarity is invariant. Given the observed action-shuffle result and signature saturation, the extra predictor-derived signature no longer justifies its complexity as the success criterion.

Keep action-probe signatures only as optional diagnostics of predicted-future equivalence, not as the online/HER hit definition.

### Worker goal-conditioning initialization

The continuous-goal FiLM adapter is currently zero initialized, so fresh runs begin exactly goal independent. For the next run, compare at least one nonzero structured initialization.

Preferred minimal options:

1. **Small orthogonal FiLM initialization:** keep the existing FiLM architecture but initialize the continuous goal adapter with a small orthogonal gain instead of exact zero. Prefer a very small gain so the parent state pathway is not disrupted.
2. **Relation decoder:** because current state and goal already share z-space, feed explicit relation features such as $[z_t,z_g,z_g-z_t]$ to a small shared decoder, optionally retaining depth/context bypasses. This is more natural for continuous same-space goals than target-ID FiLM.

If shared projections are used, apply the **same** projection to $z_t$ and $z_g$. Do not initialize independent state/goal orthogonal maps, because that would destroy their coordinate correspondence.

### Action-conditioning diagnosis

A near-zero action-shuffle delta means the current predictor behaves approximately like

$$
P_\phi(u_{t+h}\mid z_t,a_{t:t+h-1})
\approx
P_\phi(u_{t+h}\mid z_t).
$$

This can happen because the behavior policy is highly state-determined, so the executed actions add little conditional information beyond $z_t$. It does not imply that z is useless.

Log normalized predictive gains in addition to raw losses:

$$
G_z=
\frac{L_{\mathrm{state\ shuffle}}-L_{\mathrm{normal}}}
{\max(L_{\mathrm{state\ shuffle}},\epsilon)},
$$

$$
G_a=
\frac{L_{\mathrm{action\ shuffle}}-L_{\mathrm{normal}}}
{\max(L_{\mathrm{action\ shuffle}},\epsilon)}.
$$

If $G_z$ grows while raw loss grows, interpret the representation relative to the changing prediction problem rather than from raw loss alone.

Do not force action dependence merely to make `action_shuffle_delta` positive. If action-conditioned predictive state remains scientifically important, first improve action coverage/balancing in the auxiliary replay rather than adding an arbitrary action-sensitivity penalty.

## Immediate fixes

1. **Contextual HER success semantics.** DG ID is only the landmark slot/address. A HER goal selected from future raw CA3 state $S_g$ must be achieved with the same contextual recognition semantics as online goals. Keep the slot as bookkeeping, but do not let slot equality define success. Following the 23 September diagnosis, the canonical semantics use direct normalized z-space similarity:

$$
\operatorname{hit}(S_t,S_g)
=
[\operatorname{has\_DG\_event}(S_t)]
[\cos(\widehat z_t,\widehat z_g)\ge\tau_z].
$$

2. **Restore CA3-readout anti-collapse regularization.** The implementation currently has the prediction loss but omitted the planned variance/covariance terms. Restore

$$
\mathcal L_{\mathrm{state}}=\mathcal L_{\mathrm{pred}}+\lambda_{\mathrm{var}}\mathcal L_{\mathrm{var}}+\lambda_{\mathrm{cov}}\mathcal L_{\mathrm{cov}}.
$$

   initially $\lambda_{\mathrm{var}}=0.1$ and $\lambda_{\mathrm{cov}}=0.01$. Keep state/action shuffle deltas as diagnostics of whether the predictor actually uses $z$ and actions.

3. **Task-general transfer mode.** Preserve the historical DG/policy transfer scopes, but add a mode that transfers the task-general learned system: DG representation, $W$, innovation predictor, contextual anchors/graph, universal worker/controller and their normalization/state. Leave task-specific external-reward bindings/heads fresh. Keep the action interface matched unless action remapping is the explicit experiment.

## Future refinement: counterfactual probe bank

The repeated-single-action probes are acceptable for V1. If contextual signatures are unstable or insensitive, replace them with a fixed shared bank of behavior-supported replay action prefixes at several horizons. Every state must still be queried with the same probe bank; the purpose is only to reduce out-of-distribution counterfactual sequences.

## Future refinement: recognition threshold calibration

Keep positive-only calibration as V1 because within-occurrence pairs are trusted positives, whereas apparent negatives can be false negatives under route invariance or redundant DG codes.

Add background/negative diagnostics without negative training:
- cross-anchor pairs;
- temporally separated occurrences of the same DG slot;
- different-DG occurrences;
- offline coordinate-verified different physical locations.

Report positive/background similarity distributions, overlap or ROC-style diagnostics, and false-positive estimates where privileged offline labels are available. Add explicit negative training only if these diagnostics show that positive-only calibration cannot separate aliases.

## Future refinement: anchor update semantics

Separate same-identity refinement from semantic reset. Confirmed same-landmark observations may refine the anchor while preserving graph edges and generation. Clear incident graph evidence only when there is evidence that the slot identity itself changed.

Use a clean two-condition comparison:

1. **FIXED_ANCHOR.** Keep the first qualified raw CA3 anchor $A_j$ unchanged after activation. This is the stability control.
2. **EMA_SIGNATURE_ANCHOR.** Keep $A_j$ as one actually observed raw CA3 state, but maintain an exponential-moving-average predictive-signature prototype

$$
\mu_j\leftarrow(1-\alpha)\mu_j+\alpha\sigma(S_t).
$$

   over confirmed same-landmark occurrences. Do not average raw CA3 states. A confirmed candidate $S_t$ may replace $A_j$ only when its predictive signature is closer to $\mu_j$ than the incumbent's by a small margin or sustained criterion. Such refinement preserves graph edges and anchor generation because semantic identity has not changed.

This gives gradual prototype refinement without ever commanding a synthetic/nonexistent CA3 state. A true semantic reset remains a separate rare operation that increments generation and clears incident graph evidence.

The intended semantics remain one landmark per DG slot; this is not a contextual-clone proposal.

## Future refinement: simultaneous DG activations

Preferred rule if it remains a small local change:

1. collect all currently active and selectable DG slots;
2. evaluate contextual recognition for each active slot;
3. accept the landmark if exactly one active slot passes its contextual threshold;
4. if zero or multiple slots pass, mark the event ambiguous and do not update landmark/graph evidence.

This is less restrictive than raw exclusivity: several DG units may be active, but one unambiguous contextual landmark can still be recognized.

Implementation priority:
- **preferred:** unique contextual match, because the existing action-probe signature and recognition threshold can be reused and the current contextual_activity path already computes the needed score;
- **fallback if the patch ceases to be local/simple:** strongest/dominant DG activation, matching the historical visit_direct convention;
- **later only if abstention is excessive:** best contextual score with an explicit margin over the runner-up.

Keep this choice configurable so strongest-DG and unique-contextual-match can be compared directly.

Historical context: legacy visit_direct used dominant/argmax recognition. Later frontier_direct/frontier_waypoint and the frozen DDQN/HER parent used exclusive-positive landmark recognition. Treat this as a precision-versus-coverage ablation rather than assuming either rule is universally best.

## Downstream-agent implementation handoff

Implement this on top of the deployed CA3 predictive active-goal branch. Keep the current production matrix intact; do not mutate running jobs. The corrected follow-up should be a fresh namespace/checkpoint lineage.

### A. Urgent correctness fixes — required in every follow-up arm

#### A1. Make HER goal identity contextual, not DG-ID-only

Current bug: stored HER passes the raw CA3 endpoint to the worker as `virtual_goal_state`, but `controller_stored_replay.evaluate_pairs()` still defines HER start/hit from the DG slot only. This defeats the purpose of the CA3-state goal.

Canonical semantics for `ca3_worker_goal_mode=state_readout` + contextual goals:

- `virtual_goal_state` (raw CA3 snapshot) is the semantic HER goal.
- `virtual_goal` remains only a slot/address/bookkeeping field needed by existing replay/Q interfaces. It must not define success.
- A contextual HER goal is achieved by comparing the current/successor CA3 state directly with the stored goal CA3 state under the same predictive-signature metric and calibrated recognition threshold used online.
- Do not silently fall back to DG-ID success when calibration is unavailable.

Factor the similarity calculation so online graph recognition and HER call the same helper. Prefer a helper in `ca3_state_readout.py`, conceptually:

    latent_contextual_similarity(readout, left_ca3, right_ca3)

which computes cosine similarity between normalized current readout vectors `W S`. Keep the predictor-derived action-probe signature available only for diagnostics.

For stored HER:

    start_hit = has_DG_event(S_t) and cos(norm(W S_t), norm(W S_g)) >= tau_z
    next_hit  = has_DG_event(S_{t+1}) and cos(norm(W S_{t+1}), norm(W S_g)) >= tau_z

where $S_g$ is `virtual_goal_state` and $\tau_z$ is the z-space recognition threshold. Do NOT additionally require `j_t == virtual_goal` in the state-readout contextual mode. Preserve the old ID-only logic exactly for `target_id` modes.

`hindsight_examples()` should continue to choose a real future replay state and store its raw CA3 snapshot. For the contextual mode, a future endpoint only needs a real worker state and at least one DG event. If multiple DG units are active at that endpoint, use the strongest active DG only as the temporary slot/address; the endpoint CA3 state remains the actual goal identity.

Terminal contract: do not fabricate contextual CA3 from the zero-filled terminal successor used by stored replay. A terminal state may be a contextual HER target only if a certified real successor CA3 is available. Otherwise exclude/reject that contextual HER transition/endpoint with an explicit reason. The current state-readout path already does not add `terminal_dg` as a HER goal; keep that conservative behavior.

Also fix the `her_start_already_achieved` check: in contextual mode it must use the contextual goal comparison, not `canonical(core,row)[virtual_goal] > 0`.

Required regression tests:

1. Two physically/contextually different CA3 states share the same active DG slot: goal $=A$, successor $=B$. ID matches but contextual similarity is below threshold -> HER hit must be false.
2. Goal $=A$, successor $=A$-like contextual state -> HER hit true.
3. Current state has same DG slot as goal but wrong context -> must NOT reject as `her_start_already_achieved`.
4. `target_id` mode preserves the historical ID-only behavior.
5. Missing/unqualified contextual calibration never falls back to ID-only success.

#### A2. Restore the planned variance/covariance anti-collapse loss

Add two config coefficients in `custom_params.py`:

    --ca3_state_readout_var_coeff=0.1
    --ca3_state_readout_cov_coeff=0.01

Implement the already-planned VICReg-like terms in `ca3_state_readout.py`. For valid latent states $z$:

    zc = z - mean(z, dim=0)
    C  = zc.T @ zc / max(B-1, 1)
    L_var = mean_i relu(1 - sqrt(C_ii + eps))^2
    L_cov = sum_{i!=j} C_ij^2 / (d_z (d_z-1))

and train with

$$
\mathcal L_{\mathrm{state}}=\mathcal L_{\mathrm{pred}}+\lambda_{\mathrm{var}}\mathcal L_{\mathrm{var}}+\lambda_{\mathrm{cov}}\mathcal L_{\mathrm{cov}}.
$$

Compute var/cov over valid current CA3 states, not the horizon-expanded prediction-window rows if practical, so early states are not overweighted merely because they generate more horizons. CA3 input remains detached: this loss owns $W$/readout only, not DG/CA3.

Extend `ReadoutPrediction` and learner telemetry with at least:

- `ca3_readout_var_loss`
- `ca3_readout_cov_loss`
- per-dimension latent standard-deviation summary, at least mean/min

Keep the existing state/action shuffle diagnostics. Var/cov prevents collapse; state-shuffle delta tests whether the predictor actually uses $z$. These are different checks.

Required tests:

1. Constant $z$ gives positive variance penalty.
2. Unit-variance decorrelated synthetic $z$ gives near-zero var/cov penalties.
3. Loss backpropagates into $W$ but not detached CA3/DG.
4. Zero/one-sample valid batches are finite and do not NaN.

### B. Useful follow-up changes — keep small and configurable

#### B1. Fixed anchor versus EMA-signature-refined anchor

Keep legacy `champion` for checkpoint/config compatibility, but do not use it in the new primary follow-up matrix. Add an `ema` anchor mode alongside existing `fixed`.

Both modes always command a REAL observed raw CA3 anchor $A_j$; never command an averaged CA3 history.

FIXED:

- first qualified/confirmed $A_j$ remains unchanged;
- graph identity/generation and edges remain stable.

EMA:

- store one real raw CA3 anchor $A_j$ exactly as today;
- maintain a normalized EMA predictive-signature prototype $\mu_j$ from confirmed same-landmark observations;
- default starting values for the first batch: $\alpha=0.05$, minimum 8 confirmed observations before refinement, cosine-improvement margin $0.01$; keep all three configurable.

Conceptually:

    s = normalize(signature(S_candidate))
    mu_j <- normalize((1-alpha) mu_j + alpha s)

After the minimum confirmation count, compare current-model signatures:

    q_candidate = cos(signature(S_candidate), mu_j)
    q_anchor    = cos(signature(A_j),        mu_j)

and if

$$
q_{\mathrm{candidate}}>q_{\mathrm{anchor}}+\delta_{\mathrm{anchor}}.
$$

replace only the stored raw anchor:

$$
A_j\leftarrow S_{\mathrm{candidate}}.
$$

This is SAME-IDENTITY REFINEMENT. It must NOT:

- clear graph edges;
- increment anchor generation;
- deactivate the goal;
- reset visit/control evidence.

Reserve the existing destructive reset/invalidation path for true semantic reassignment of a DG slot, not ordinary prototype refinement.

Representation-drift caution: $W$ and predictor continue learning, so a signature EMA contains samples from slightly different representation snapshots. Keep the EMA local/recent. When `recalibrate()` changes the recognition model/threshold epoch, it is acceptable for V1 to reinitialize each valid $\mu_j$ from the CURRENT signature of its real anchor and then resume EMA updates. Do not store a long-lived stale prototype across large readout drift without a test.

Checkpoint all new EMA buffers/counters and cover exact reload.

Useful telemetry:

- anchor refinements (non-destructive)
- mean anchor age
- mean candidate-vs-anchor centrality gain
- per-node confirmation count
- destructive semantic resets separately from refinements

#### B2. Multi-activation recognition: unique contextual match, with dominant fallback

Add a config such as:

    --ca3_context_candidate_mode={exclusive,dominant,unique_contextual}

Preserve `exclusive` for legacy reproduction.

`dominant`: choose the strongest positive DG activation, then require that slot's contextual anchor similarity to pass $\tau$. If it fails, emit no landmark.

`unique_contextual` (preferred if the patch remains local):

1. collect every active DG slot that is currently selectable;
2. compute contextual similarity to that slot's anchor;
3. retain candidates with similarity $\ge\tau$;
4. if exactly one candidate passes, recognize it;
5. if zero or >1 pass, emit no landmark/graph event.

This permits multiple raw DG activations while still demanding one unambiguous semantic landmark. It reuses the existing anchor/signature/threshold machinery and should stay inside `PolicyControllableGraph.contextual_activity()`.

Do not add a runner-up margin in this batch. Only add best-score-plus-margin later if the unique rule abstains too often.

Required tests:

1. two raw DG units active, only one contextual match -> recognize that one;
2. two active and both pass -> abstain;
3. strongest raw activation fails but weaker activation is the sole contextual match -> `unique_contextual` recognizes the weaker one, while `dominant` abstains;
4. no active selectable slot -> no hit;
5. legacy `exclusive` behavior unchanged.

Telemetry needed to interpret this factor:

- raw multi-activation fraction;
- fraction of multi-active observations rescued by a unique contextual match;
- contextual zero-match fraction;
- contextual multi-match/ambiguous fraction;
- accepted contextual-event fraction.

#### B3. Recognition threshold calibration diagnostics — diagnostic only

Do NOT add negative training in this follow-up. Keep positive-only threshold calibration unchanged for the learning mechanism.

Add diagnostics named clearly under recognition-threshold calibration. At minimum record/compute:

- positive similarity $q_{10}/q_{50}/q_{90}$;
- cross-anchor/background similarity $q_{50}/q_{90}/q_{99}$;
- fraction of background comparisons above the current $\tau$;
- pairwise active-anchor collision fraction;
- offline coordinate-verified false-accept rate where pose is available only to the evaluator.

Background samples may include cross-anchor pairs, temporally separated same-DG occurrences, and different-DG occurrences. Treat these as diagnostics, not automatically valid negatives.

#### B4. Improve shuffle diagnostic if trivial

Current `latent.roll(1,0)` can pair temporally adjacent/related windows. If easy, replace it with a permutation that breaks stream/temporal locality while preserving horizon/action bookkeeping. This is useful telemetry, not a blocker for the corrected batch.

### C. Follow-up batch: small $2\times2$ factorial

Use H32 for the follow-up because the current production batch already contains the H16/H32 comparison and H32 is the intended long-context condition. Do not duplicate the seven-arm architecture sweep.

All four arms MUST share:

- contextual HER fix A1;
- var/cov fix A2;
- `ca3_state_readout_mode=worker`;
- `ca3_state_readout_horizon=32`;
- action conditioning ON;
- `ca3_worker_goal_mode=state_readout`;
- `ca3_graph_contextual_hits=True`;
- F64, navigation-eight, stored DDQN+HER, and the same production controller/replay budgets;
- same positive calibration settings unless calibration itself is the explicit later experiment.

Factor 1 — anchor update:

- `FIXED`: `ca3_graph_anchor_mode=fixed`;
- `EMA`: new `ca3_graph_anchor_mode=ema` with the $\alpha$/minimum-confirmations/margin defaults above.

Factor 2 — simultaneous-DG candidate rule:

- `DOM`: strongest-positive DG candidate + contextual threshold;
- `UNIQUE`: unique contextual match among all active selectable slots.

Primary four cells:

1. `CTX_FIXED_DOM_H32`
2. `CTX_FIXED_UNIQUE_H32`
3. `CTX_EMA_DOM_H32`
4. `CTX_EMA_UNIQUE_H32`

Use seeds $8$, $99$, and $123$, giving $12$ production runs. This directly estimates:

- EMA refinement effect averaged over recognition rule;
- unique-context recognition effect averaged over anchor rule;
- whether their interaction matters.

Do not add `champion` as a fifth primary cell unless resources are abundant; the running September-22 production already documents that legacy mechanism, and the corrected fixed anchor is the cleaner stability control.

Mechanical release gate: one short qualification run per cell (seed $99$ is sufficient) must verify finite loss/telemetry, contextual HER actually exercises true/false contextual hits, calibration becomes ready, EMA buffers checkpoint/reload when applicable, and multi-active recognition paths are exercised. This gate is for correctness only, not condition selection. Prepare all $12$ production configs in parallel; release them after the four-cell mechanical gate passes.

### D. Required interpretation metrics for the follow-up

Keep the existing coverage/grounded-controllability/controller outcomes, and add the following because they directly test the new fixes:

Representation:

- prediction loss, active/zero loss;
- var loss, cov loss;
- latent std min/mean;
- state-shuffle delta, action-shuffle delta.

HER semantics:

- contextual HER candidate count;
- contextual HER positive-hit count/rate;
- contextual HER rejected-same-DG-wrong-context count;
- `her_start_already_achieved` contextual count;
- missing-calibration/missing-successor rejection counts.

Recognition:

- accepted contextual-event rate;
- raw multi-activation rate;
- unique-match rescue rate;
- zero-match and multi-match ambiguity rates;
- recognition $\tau$ and positive/background calibration diagnostics.

Anchor stability:

- registrations;
- confirmations;
- non-destructive refinements;
- destructive resets/replacements;
- active-goal count;
- graph evidence lost due to destructive reset (should be zero for ordinary EMA refinement).

### E. Task-general transfer mode — implement in parallel, do not block the no-reward follow-up

The existing `transfer_scope={none,dg,policy}` predates this architecture. Add a `task_general`/`world_model` transfer mode for the downstream physical-reward experiment.

Design principle: load the whole compatible task-general navigation system and explicitly reset task-specific state, rather than maintaining another fragile allow-list that silently omits new modules.

For a matched navigation-eight downstream task, retain at least:

- DG projection + BN state;
- fixed CA3 configuration;
- state readout $W$;
- innovation predictor;
- contextual anchors, recognition calibration, active masks and graph fast weights;
- worker decoder/FiLM;
- goal-conditioned controller Q/navigation policy.

Reset/do not inherit:

- optimizer state and training progress;
- replay buffer and episode/RNN transient state;
- exploration/annealing counters unless explicitly part of the test;
- downstream external-reward binding/head/value state that is task-specific.

Initialize any target-Q copy from the transferred online-Q state rather than importing an arbitrary stale lag unless the transfer experiment explicitly studies optimizer/target-state continuation.

Keep the action interface identical to pretraining for the first transfer experiment. The historical five-action fixed-reward pilot is not a clean test of this new task-general world model.

Add an inventory test: a newly added task-general module must fail loudly if omitted from `task_general` transfer, instead of being silently left fresh.

### F. Explicit non-goals for this patch

- Do not redesign the H16/H32 predictor or probe bank now.
- Do not add negative contrastive training for recognition.
- Do not add contextual clones/multiple semantic landmarks per DG slot.
- Do not re-enable DG orthogonal recruitment for this batch.
- Do not let controller/HER gradients enter $W$; preserve the current separated objective.
- Do not change graph planning, frontier scoring, or reward scale while testing these fixes.

### G. Completion criteria before launch

The downstream agent should consider the patch ready only when:

1. contextual HER no longer uses DG-ID equality for state-readout goal success/start-achieved logic;
2. var/cov losses are finite, logged, and update $W$ only;
3. fixed and EMA anchor modes checkpoint/reload exactly;
4. EMA refinement preserves graph evidence/generation;
5. dominant and unique-context candidate modes pass the multi-activation unit tests;
6. legacy target-ID/exclusive configurations remain reproducible;
7. all four follow-up StudySpec cells parse and pass the short runtime/reload audit;
8. no running September-22 production namespace/checkpoint is modified.


## 23 September next CPU/GPU launch

The next release isolates the two suspected bottlenecks while holding H32, EMA anchors, dominant DG candidate selection, DDQN+HER, F64 capacity, and the existing frontier-waypoint manager fixed.

### Scientific matrix

Run the 2x2 factorial concurrently:

| Recognition | Worker decoder | Meaning |
|---|---|---|
| `PROBE` | `FILM` | source-matched legacy control |
| `Z` | `FILM` | direct-z success semantics only |
| `PROBE` | `REL` | relation decoder only |
| `Z` | `REL` | full proposed fix |

`Z` means contextual online/HER success is cosine similarity between normalized current and goal readouts $W S_t$ and $W S_g$.

`REL` means the worker receives an ordinary MLP relation interface built from $[z_t,z_g,z_g-z_t]$ plus the existing non-CA3 bypass/context features. It removes the exact zero-goal symmetry of the continuous-goal FiLM path.

Do not add another anchor/candidate factorial in this batch.

### Source

Source branch:

`xiaoxionglin/SF_hipposlam:codex/ca3-state-goal-followup-20260922`

Code/reference commit for the fix:

`d9d8ce6653a95be718814bf226c44e852df931ea`

The implementation preserves legacy modes through:

- `--ca3_context_similarity_space={probe,z}`;
- `--ca3_worker_decoder={film,relation}`.

Action-probe signatures remain available for diagnostics and EMA-anchor refinement; they no longer need to define semantic success in `Z` arms.

### StudySpecs

CPU:

`hpc_runs/studies/ca3_zrelation_followup_20260923_cpu.study.json`

- 4 cells;
- seed 99;
- 50M frames;
- early checkpoints at 2M and 5M;
- later checkpoints at 25M and 50M.

GPU/G500:

`hpc_runs/studies/ca3_zrelation_followup_20260923_gpu.study.json`

- 4 cells x seeds 8, 99, 123 = 12 runs;
- 150M frames;
- checkpoints at 2M, 5M, 25M, 75M, and 150M.

Do not wait for a separate scientific preflight. The 2M checkpoint is the early stop point for implementation failures.

### CPU validation and submission

From the synchronized source checkout:

~~~bash
python -m hpc_runs.intrmotiv_study validate \
  hpc_runs/studies/ca3_zrelation_followup_20260923_cpu.study.json

python -m hpc_runs.intrmotiv_study render-runs \
  hpc_runs/studies/ca3_zrelation_followup_20260923_cpu.study.json \
  --output /tmp/ca3_zrelation_cpu_runs.json

sf_working_directories/IntrMotiv/launcher/launch_nemo2.sh \
  sf_working_directories.IntrMotiv.dmlab.experiments.ca3_zrelation_followup_cpu \
  --print-only

sf_working_directories/IntrMotiv/launcher/launch_nemo2.sh \
  sf_working_directories.IntrMotiv.dmlab.experiments.ca3_zrelation_followup_cpu \
  --submit
~~~

Audit the resulting `jobs.tsv` with the exact StudySpec fingerprint.

### G500 validation and direct launch

Synchronize the same reviewed source commit into a fresh G500 source directory, for example:

`/scratch/lin/IntrMotiv/src/SF_hipposlam_ca3_zrelation_20260923_d9d8ce6`

Then review:

~~~bash
python -m hpc_runs.intrmotiv_study validate \
  hpc_runs/studies/ca3_zrelation_followup_20260923_gpu.study.json

python hpc_runs/hosts/g500/launch_study.py \
  hpc_runs/studies/ca3_zrelation_followup_20260923_gpu.study.json \
  --source /scratch/lin/IntrMotiv/src/SF_hipposlam_ca3_zrelation_20260923_d9d8ce6 \
  --gpus 0 1 0 1
~~~

After reviewing the emitted direct-manifest SHA, execute the exact reviewed manifest:

~~~bash
python hpc_runs/hosts/g500/launch_study.py \
  hpc_runs/studies/ca3_zrelation_followup_20260923_gpu.study.json \
  --source /scratch/lin/IntrMotiv/src/SF_hipposlam_ca3_zrelation_20260923_d9d8ce6 \
  --gpus 0 1 0 1 \
  --execute <REVIEWED_MANIFEST_SHA>
~~~

### Early 2M stop checks

Do not require scientific success at 2M. Only stop for correctness/mechanism failures:

1. z-space online and HER hit paths disagree;
2. relation decoder is goal-insensitive at initialization or runtime;
3. raw CA3 leaks into the learned-state worker path;
4. RL/HER sends gradient into $W$ or prediction sends gradient into DG;
5. replay/checkpoint reload fails;
6. any traceback, NaN, stale-generation contract failure, or missing W&B telemetry.

At 5M and beyond compare:

- `option_success_fraction`;
- correct-goal versus shuffled-goal hit lift;
- goal/action sensitivity;
- contextual HER positive rate;
- recognition threshold and z-space positive/background similarity quantiles;
- state-shuffle and action-shuffle deltas;
- coverage AUC and frontier reach/discovery yield.

The central test is whether `Z_REL` raises option success and target-specific behavior substantially earlier than the current probe+zero-FiLM system.
