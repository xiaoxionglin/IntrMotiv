# Encoder geometry and intrinsic control: what the loss is missing

14 September 2026. Mathematical scrutiny and a proposed experimental mechanism;
no new runtime, StudySpec, training matrix, or submission. This continues the
[oracle discussion](oracle_dg_place_fields_20260914.md) and
[control-representation principle](control_representation_principle.md).

## Diagnosis and evidence boundary

Externally rewarded navigation working is evidence that the representation,
memory and controller can support a useful task under that training signal.
It does not isolate the added encoder loss as the sole cause of IntrMotiv’s
failure. IntrMotiv also changes the reward’s grounding, goal vocabulary,
conditioning, visitation distribution, termination and credit assignment.
The R3 live audit recovered successful scores/configurations, not its complete
archived learner; “pure joint training” remains the user’s historical account
rather than a new code-level verification in this task.

The stronger defensible diagnosis is that the current encoder objective does
not directly encode the desired geometric or control property, while the
intrinsic success event is itself defined by the changing encoder. An encoder
can learn a useful input to an externally grounded controller without each
individual DG unit being a coherent, stable goal detector. These are different
representation requirements.

The old loss catalogue contains pre-JOINT gradient-routing descriptions. For
this discussion use the dated DGP design, its parent StudySpec, and the retained
September 7 runtime snapshot for the reward-credit path. Do not reinterpret
JOINT as clearing PPO gradients into DG. The later matched CPD results also
show that adding a predictor/context feedback has not established a reliable
place-field or control improvement; another prediction head alone is not an
adequate answer.

## What the actual encoder update can and cannot do

Write the ARR encourage component schematically, suppressing valid-sample
normalization and action/observation indexing offsets:

$$
\mathcal L_{ARR}=-\mathbb E_t\left[\beta\,\operatorname{sg}(d_t)
\sum_j\operatorname{sg}(m_{tj})a_{\theta,j}(x_t)\right].
$$

Here $m$ is the accepted dominant onset/local-predecessor credit mask, $d$ is
the measured CA3 event interval, and `sg` means stop-gradient. Consequently,

$$
\nabla_\theta\mathcal L_{ARR}=-\mathbb E_t\left[
\beta d_t\sum_jm_{tj}\nabla_\theta a_{\theta,j}(x_t)\right].
$$

Elapsed time determines the strength of an activation update; it does not
supply a derivative of travel distance or a negative example at a competing
physical location. Replacing $d$ by $d-R$ adds a temporal sign change at credited
onsets, not a spatial margin. BatchNorm and row renormalization couple samples
and can indirectly suppress responses elsewhere, but they do not identify
which suppression is behaviorally desirable.

With frozen normalization and a linear active unit, the simplified update
rotates its row toward the reinforced feature. The change at another feature
depends on feature similarity, not manifold travel distance. Two distant views
sharing that visual direction can both be strengthened. Row orthogonality
changes directions in ambient feature space; it imposes neither connected
fields nor distinguishable action consequences.

This reward-weighted activation update is also not automatically an unbiased
policy gradient for a deterministic encoder. PPO provides a separate route
through its action probabilities in JOINT. Neither route differentiates through
the simulator’s clock or through future physical observations.

## The geometry problem is partly architectural

For a frozen observation embedding $v=F(O(x,y,r))$, a DG unit has an active
region approximately equal to a half-space intersected with the empirical
visual manifold (normalization changes the effective threshold/normal).
That intersection may have disconnected components.

A concrete smooth, injective counterexample is the embedded circle

$$
v(t)=(\cos2t,\sin2t,\epsilon\cos t,\epsilon\sin t),\quad
t\in[0,2\pi),\quad\epsilon>0.
$$

The unit selecting the first coordinate and thresholding at 0.8 fires near
both $t=0$ and $t=\pi$. The final two coordinates make the embedding injective;
the multiple fields are therefore possible even without exact perceptual
aliasing. A single current DG row is not guaranteed to express an arbitrary
compact target neighborhood. The encoder cannot move the frozen input manifold;
it changes its projection/partition of that manifold.

In the real task, the manifold description is itself an approximation:
occlusions, repeated scenes, depth compression and heading-dependent images
can create folds or aliases. If identical features require different actions,
no visual-only deterministic DG loss can separate them; history is needed.
Heading should not automatically be removed—it matters for action choice.
Position-only map localization and a sufficient control state are distinct.

The desired criterion is not “push every pair apart.” It is to retain distinctions
needed for different controlled outcomes while preserving local continuity and
useful invariances. Some multi-field codes can be sufficient when CA3 resolves
their ambiguity, but they may still be unsuitable as standalone goal IDs.

## How elapsed time can provide differentiable supervision

Do not try to differentiate elapsed wall-clock/decision count. Differentiate a
representation or predictor trained from transitions and time labels.
For an experienced path of $k$ unit-cost decisions, $k$ is an upper bound on
the shortest observed-path cost between its endpoints, not proof that they are
far apart. A long loop can end exactly where it began. Thus regressing latent
distance to every observed $k$, or treating every temporally remote pair as a
negative, produces contradictory targets on revisits.

A useful local geometric contract is

$$
\|u_\theta(x_{t+1})-u_\theta(x_t)\|_2\leq1
$$

on actual within-episode transitions. By the triangle inequality, if the bound
holds on every edge of a path,

$$
\|u_\theta(x_{t+k})-u_\theta(x_t)\|_2\leq k.
$$

If all relevant edges are covered, take the minimum over paths for a shortest-
path bound. This is **an upper bound**, not an equality or a guarantee that far
states are separated. It prevents arbitrary local stretching; an expansion
objective must supply the other pressure. Finite-data penalties only approximate
the contract. In stochastic/asymmetric environments, a symmetric latent norm
does not equal directed expected hitting time.

## Recommended mechanism: command-specific progress under a temporal constraint

The closest established basis is [METRA](https://arxiv.org/abs/2310.08887),
whose joint skill/representation objective uses latent progress with a
transition-based distance constraint. [LSD](https://arxiv.org/abs/2202.00914)
is an earlier related approach using a different metric constraint. These are
prior art, not a claim of a new algorithm. The following is an IntrMotiv
adaptation to investigate, not a proven repair or an exact METRA reproduction.

The first hypothesis should use the existing 32-dimensional DG activity as
$u_\theta$ so that the reward cannot live entirely in an auxiliary representation
the worker never observes. The worker still receives its CA3 history and the
existing depth/instruction bypass. Keep the reward encoder goal-independent;
do not feed skill ID, episode ID, elapsed clock, or CA3 into $u$. Otherwise it
could report progress from an internal command/timer instead of environmental
change. Current sparse ReLU dead zones and normalization are explicit feasibility
risks, not reasons to silently add a bypass encoder.

Sample a command $g$ uniformly from $K$ channels independently of the initial
state, with centered direction

$$
c_g=\frac{e_g-\frac1K\mathbf1}{\|e_g-\frac1K\mathbf1\|_2}.
$$

For the cleanest first formulation, retain that command for a fixed-length
physical episode. This is a skill direction associated with a DG channel,
not an independently grounded destination or a manager-certified option.
Define

$$
r^{int}_t=\operatorname{sg}\left[
c_g^\top(u_\theta(x_{t+1})-u_\theta(x_t))\right],
$$

and optimize the constrained encoder objective

$$
\max_\theta\mathbb E\left[c_g^\top(u_\theta(x_{t+1})-u_\theta(x_t))\right]
\quad\text{subject to transition locality above.}
$$

PPO maximizes the same progress through actions. The encoder updates both
endpoints on recorded transitions; the simulator is not differentiated.
The channel-specific derivative reinforces the commanded endpoint relative to
its start and relative to other channels. Unlike raw activation reinforcement,
uniformly increasing that channel at every state contributes no progress.
The finite local-change budget prevents unrestricted feature-scale inflation
on the transitions where the constraint is actually satisfied.

This objective replaces ARR interval-weighted reinforcement and the current
worker HIT reward in the experimental arm. Do not add it atop all existing
repulsion/margin/predictor terms. Keep only declared existing resource-maintenance
terms needed to avoid permanently dead DG rows, and report their separate
contribution. A constraint solver/penalty and its violations must be qualified;
a soft average penalty is not a hard pointwise guarantee. This is a change to
the coupled objective and command semantics, not a drop-in encoder regularizer.

### Useful properties under stated assumptions

For a fixed encoder, fixed command and undiscounted finite horizon $T$,

$$
\sum_{t=0}^{T-1}r_t=c_g^\top(u(x_T)-u(x_0)).
$$

An exact loop has zero net progress, and a stationary observation has zero
reward. If the trajectory distribution ignores $g$, balanced independent
commands imply zero expected signed progress. Different commands have to change
outcomes to improve that expectation. These are properties of the defined
objective, not guarantees of optimization or exploration.

The telescoping argument does **not** hold unchanged for the parent’s
$\gamma=0.99$ discounted return. An out-and-back path can receive positive
discounted return when outward reward arrives first. For a finite-episode
mechanism test use $\gamma=1$ with correct physical terminal handling, or
derive and test an explicit terminal objective; do not claim loop immunity for
an unmodified discounted learner. Fixed-horizon terminal endpoint value must
be retained. A continuing discounted potential-shaping term with no external
reward is not an automatic solution.

Freeze the reward representation and its normalization over each collection/
PPO update block, store behavior rewards, and keep continuous event identities
consistent on both endpoints. A detached scalar alone does not freeze future
reward definitions. Block freezing reduces inconsistency but is not permanent
semantic grounding. Record endpoint drift under a fixed observation panel.
At physical reset, never treat the reset observation as the preceding action’s
outcome; obtaining the real terminal observation is a preflight requirement.

### Limits that remain

- Sparse DG may make gradients weak or saturate progress. Measure whether the
  encoder objective reaches the same channels the worker receives. Compare
  with a richer metric representation only as a separately named architecture
  factor if the linear readout is demonstrably inadequate.
- Turning in place may be a controllable visual change. A temporal metric
  cannot promise translation, position-only fields, or broad coverage.
- Common visual aliases, partial observability, unobserved shortcuts, policy-
  dependent data and stochastic distractors remain limitations.
- The representation can still deform between updates. The constraint limits
  some distortions; it does not give fixed meanings to endogenous goal IDs.
- No guarantee of connected fields, all-goal reachability, skill reliability,
  or universal transfer follows from a positive objective.

If strict compact goal fields are needed, a later architectural alternative
is localized prototype readout in a qualified metric space, e.g.
$a_j=[1-\|u-p_j\|^2/\rho^2]_+$. Even a compact latent ball can pull back to
multiple physical regions if $u$ folds the manifold. Prototype stability,
coverage and metric fidelity need evidence. Do not claim that an RBF alone
solves controllability, or add prototypes before testing the objective.

## Closest alternative when fixed landmark goals must be preserved

Use observed future sensory exemplars as stable outcomes and learn an
action-conditioned contrastive reachability critic through DG/CA3.
[Contrastive RL](https://arxiv.org/abs/2206.07568) provides a principled link
between such contrasts and goal-conditioned values under its sampling and
model assumptions. Its score is not automatically a calibrated hitting time,
and a generic contrastive head added to the existing moving DG labels does not
inherit that result. This project already has off-policy goal baselines; consult
them before building another implementation.

This route is closer to supplied-goal control with self-generated targets;
the progress route tests intrinsic skill discovery more directly. A further
action-conditioned prediction head on endogenous DG outcomes would repeat the
CPD ambiguity: predictable labels can still be useless or collapse. The same
frozen outcome set and commanded-versus-null evaluation remain essential.

## What should be tested before another production batch

First distinguish three questions, without committing to a large factorial:

1. **Is the current encoder objective harmful?** In one otherwise matched
   IntrMotiv configuration compare JOINT with ARR on versus ARR off, retaining
   the same maintenance terms. This isolates that loss’s effect but does not
   solve moving success labels or prove a new reward useful.
2. **Can the new objective satisfy its own contract?** Use a small controllable
   graph and then a fixed visual observation/transition panel. Include loops,
   return visits, heading changes, aliases and resets. Check local-constraint
   violations, collapse, command-independence null, and the actual learner’s
   return on a closed loop. Do not repeat the earlier threshold-rotation toy:
   it had no CA3, temporal-distance labels, or learned control.
3. **Does it generate usable control?** Only after those gates, run a bounded
   matched-seed environment pilot. Use matched starts and commands, fixed-panel
   representation checks, movement/coverage, and independently scored physical
   outcomes. Separate intrinsic pretraining quality from downstream fixed-reward
   transfer. External reward success is a positive control, not interchangeable
   with intrinsic reward magnitude.

These are proposed diagnostic contrasts, not a declarative run matrix. Any
execution plan must use the current canonical StudySpec workflow and retain
all source/checkpoint/config fingerprints. Do not reinterpret earlier oracle
plan approval as authorization to train this different algorithm.

## Evidence and reusable lessons

Local sources: [loss catalogue](../04_implementation/architecture/losses.md),
[DGP design](joint_ppo_dg_first_outcome_batch.md),
[DGP failure audit](../06_experiments/dgp_interim_failure_audit_20260907.md),
[matched feedback results](../06_experiments/05_ca3_feedback_matched_results_and_full_state_gap_20260913.md),
[historical time-gradient note](../03_transition_distance/DG_CA3_gradient_timing_evaluation.md),
[threshold toy](../06_experiments/threshold_rotation_toy_report.md).
The scoped source archive inspected was
`hpc_runs/source_snapshots/ca3_memory_novelty_goal_20260907.tar.gz`:
`custom_learner.py` retains no-grad behavior credit construction and the
`rewards_encoder`/`encoder_credit_activation_mask` fields. This is archived
evidence of the credit mechanism, not a new audit of active NEMO2 source.

The decisive shortcut was writing the gradient and a folded-manifold
counterexample before designing another regularizer. Dated source/design
records supersede old catalogue claims about gradient clearing. Primary papers
support a constrained temporal objective and contrastive alternative, not a
guaranteed IntrMotiv fix. The arXiv HTML URLs failed; the canonical PDF exposed
the relevant equations directly. Next time check reward grounding, observed
path versus shortest-path semantics, collapse, and discounting before proposing
elapsed-time differentiation. No numerical experiment or rollout was performed
for this note; its guarantees are conditional mathematical statements.
