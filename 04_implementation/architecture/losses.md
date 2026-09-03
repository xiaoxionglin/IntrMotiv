# IntrMotiv Loss Catalogue

All learner averages below exclude invalid/padded replay entries. `F` is the
number of DG units, `z_{tj}` is the pre-threshold DG logit, `a_{tj}` is
post-threshold DG activity, and `theta` is the DG threshold. The standard
projection is

$$
a_{tj}=\max(0,z_{tj}-\theta).
$$

The `layer2_resnet18` trunk is ImageNet-pretrained and frozen. Unless noted,
DG losses update the DG projection and BatchNorm parameters/statistics, not
the ResNet.

## 1. Reward Signals, Not Losses

These are used to construct advantages or reward-weighted encoder objectives.
They are not directly differentiated through trajectory sampling.

### 1.1 Temporal-distance intrinsic signal

Let `p_t[j]` be the first occupied position of DG `j` in the CA3 register,
with absent units assigned `E=R+L-1`. A qualifying newly appearing DG event
produces temporal separation `d_t`: the nearest other current/previous sequence
position, with a sentinel value `E` when no event exists.

The legacy decoder signal is

$$
r^{dec}_t=\beta(E-d_{t+2}).
$$

The encoder receives one of:

$$
r^{enc}_t=
\begin{cases}
\beta d_{t+1} & \texttt{encourage},\\
\beta(d_{t+1}-E) & \texttt{punish},\\
\beta(d_{t+1}-\bar d) & \texttt{mean}.
\end{cases}
$$

The offsets are implementation alignment details: worker rewards index
`[:, 2:]`; encoder rewards index `[:, 1:-1]`. `d_t` is temporal sequence
separation, not Euclidean distance and not `T_ctrl`.

### 1.2 HRL worker reward

For an intended waypoint/target hit indicator `h_t`, the `hit_distance` worker
reward is

$$
r^{worker}_t=r_{hit}+\alpha\max(0,r^{dec}_t),\qquad\text{only if }h_t=1,
$$

and zero otherwise. `hit` mode omits the distance bonus. During a manager
exploration option, the bounded dense temporal-distance signal is used instead.
This reward trains the worker PPO objective; it is not a manager reward.

## 2. PPO Worker Objective

The worker uses Sample Factory PPO/APPO infrastructure with `with_vtrace=False`
in current IntrMotiv batches. GAE is computed from the selected worker reward:

$$
\delta_t=r_t+\gamma V_{t+1}-V_t,
\qquad
A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}.
$$

For `q_t=\pi_\theta(a_t\mid x_t,g_t)/\pi_{old}(a_t\mid x_t,g_t)`, the clipped
policy term is

$$
\mathcal L_{policy}=-\mathbb E[\min(q_tA_t,\operatorname{clip}(q_t,l,u)A_t)].
$$

The framework adds value regression, entropy regularization, and optional KL
penalty:

$$
\mathcal L_{worker}=\mathcal L_{policy}+c_V\mathcal L_{value}
-c_H H(\pi)+\mathcal L_{KL}+\mathcal L_{extra-decoder}.
$$

PPO gradients into `DG_projection` are explicitly cleared. Thus the worker
objective updates the decoder/action/value networks, but does not directly
train DG landmarks.

## 3. DG Reward-Weighted Encoder Objective

Let `E_t[j]` identify a DG onset whose progression is zero now and was at least
`R` on the preceding decision. If several units begin together, select the
largest behavior activation as `D_t`; exact ties use the lowest DG index. All
simultaneous candidates are omitted from the predecessor-distance search, but
only `D_t` receives the encoder update:

$$
\mathcal L_{enc-reward}=
-\mathbb E_t\left[\sum_j r^{enc}_t a_{tj}D_t[j]\right].
$$

It updates the DG projection through a separate encoder forward/backward pass.
Its sign and behavioral meaning depend on `encoder_reward_method`; do not
compare its raw magnitude across reward methods without accounting for that.
The dominant label is derived from accepted rollout CA3 state and retained in
the learner batch, rather than being reselected from newer minibatch logits.

## 4. DG Recruitment And Population Objectives

### 4.1 Batch unused-unit recruitment

For a learner minibatch, define `U_j=1` when DG `j` is absent from both incoming
CA3 and current post-threshold activity on every valid transition. Define a
smooth pre-threshold activity surrogate with temperature `T_b`:

$$
s_{tj}=T_b\operatorname{softplus}\left(\frac{z_{tj}-\theta}{T_b}\right).
$$

With `encoder_batch_loss=True`:

$$
\mathcal L_{batch}=-\mathbb E_{t\in valid}\left[
\frac{\sum_j U_js_{tj}}{\max(1,\sum_jU_j)}\right].
$$

Unlike the historical post-ReLU form, this has nonzero gradient below the DG
threshold and can recruit a silent row. `T_b` is
`encoder_batch_loss_temperature` (default `0.5`). Minimization encourages a
currently unused resource to activate. The loss is normalized by the unused
row count, is batch-scoped, and can be negative by construction. It is normally
enabled in IntrMotiv batches.

### 4.2 Optional local unused-sequence recruitment

`encoder_unused_sequence_loss` rewards a dominant onset when that unit's own
incoming CA3 chain was empty:

$$
\mathcal L_{unused}=-\mathbb E_t\left[\sum_j a_{tj}C^{unused}_t[j]\right].
$$

It overlaps conceptually with the batch loss and is normally disabled.

### 4.3 Optional multi-activation penalty

`encoder_multi_activation_loss` penalizes every non-dominant candidate when
multiple DG units begin together:

$$
\mathcal L_{multi}=\mathbb E_t\left[\sum_j a_{tj}(E_t[j]-D_t[j])\right].
$$

It is normally disabled in historical batch commands. Dominant selection alone
therefore does not imply that a running job applies this penalty.

### 4.4 Population usage, density, and collision

When `encoder_population_usage_loss=True`, define mean activity
`u_j=\mathbb E_t[a_{tj}]+\epsilon` and `p_j=u_j/\sum_k u_k`. The three terms
are:

$$
\mathcal L_{usage}=\sum_j p_j\log(Fp_j),
$$

$$
\mathcal L_{density}=(\mathbb E_{t,j}[a_{tj}]-\rho)^2,
$$

$$
\mathcal L_{collision}=\mathbb E_t[\max(0,\sum_j a_{tj}-1)^2].
$$

The combined objective is

$$
\lambda_u\mathcal L_{usage}+\lambda_d\mathcal L_{density}
+\lambda_c\mathcal L_{collision}.
$$

It distributes average activity across units, targets density `rho`, and
discourages multi-unit magnitude at a step. It is default-off; its three
coefficients are configured separately.

## 5. Anti-Collapse And Localization Objectives

### 5.1 Global pre-threshold punishment

With positive `dg_global_punishment_coeff` and temperature `T`:

$$
\mathcal L_{global}=\lambda_g\mathbb E_{t,j}
\left[T\,\operatorname{softplus}\left(\frac{z_{tj}-\theta}{T}\right)\right].
$$

This is a smooth penalty on every pre-threshold logit, including inactive DG
units. Because DG projection rows are renormalized to unit norm after each
optimizer step, the effective response is mostly rotation/reallocation rather
than unconstrained shrinking. It is default-off and requires `batchnorm_relu`.

### 5.2 DG-row angular repulsion

Let `w_j` be DG projection rows normalized to unit length. The row-only
regularizer is

$$
\mathcal L_{row}=\lambda_r\frac{1}{F(F-1)}
\sum_{j\ne k}(w_j^\top w_k)^2.
$$

It makes projections span different input directions without relying on
observations. It is default-off and does not itself create spatial fields.

### 5.3 CA3 temporal exclusion

Let `P_t[k]` denote whether DG `k` occupied the CA3 tap at `R-1` in the
stored state. That tap represents the preceding `R`-decision window; the loss
does not scan the older `L` history. For each candidate unit:

$$
M_{tj}=\left[\sum_{k\ne j}P_t[k]>0\right],
\qquad
\mathcal L_{CA3-x}=\lambda_x\mathbb E_{t,j}[a_{tj}M_{tj}].
$$

It penalizes a current DG activation when a *different* DG was recently active.
It does not penalize the same unit's continuation. `ca3_conflict_fraction` is
the mean of `M` over all unit-time slots; `ca3_conflicting_activation_fraction`
is the fraction of current active entries satisfying `M`. They use different
denominators and are not expected to have similar values.

### 5.4 Same-DG path scatter loss

This is available only for a topological manager with action path integration.
Each DG gets an episode-local anchor on its first exclusive activation. Let
`d_tj` be command-integrated displacement from that anchor and `s_t` the
current trace straightness. Its detached mask is

$$
M_{tj}=\mathrm{anchorValid}_j[d_{tj}\ge d_{min}][s_t\ge s_{min}].
$$

The loss is

$$
\mathcal L_{scatter}=\lambda_s
\frac{\sum_{t,j}M_{tj}\,T\operatorname{softplus}((z_{tj}-\theta)/T)}
{\sum_{t,j}M_{tj}\vee1}.
$$

It suppresses far straight-trace reactivation of the *same* unit. It does not
repel distinct nearby DG units and deliberately ignores loop-like paths with
low straightness. Default-off.

### 5.5 Shadow CA3 target predictor

The optional predictor receives detached CA3 state and stored worker target.
For a horizon `H`, it predicts whether that target will activate and the
normalized hit time. Its loss is

$$
\mathcal L_{pred}=\operatorname{BCEWithLogits}(\hat h,h;w_+)
+\operatorname{SmoothL1}(\hat\tau,\tau/H)\quad\text{on positive labels}.
$$

The positive weight is clipped class balancing. The input CA3/DG labels are
detached, so this trains the predictor head only. It is diagnostic/shadow
modeling: current managers do not consume its output. Default-off.

## 6. Structural Updates, Not Gradient Losses

### 6.1 Orthogonal DG recruitment

Orthogonal recruitment has two modes. `legacy` preserves the historical rule:
at an endpoint exactly `L` decisions after a lone source activation, if the
current feature activates no DG unit, one never-committed, least-used row may
be replaced. `graph` keeps the same silent-endpoint proposal but determines
which row may be replaced from landmark-graph evidence rather than the
`recruitment_committed` flag.

In graph mode, supported directed edges are

$$
A_{ij}=[C_{ij}>0.25][T_{ij}>0].
$$

A mature vertex is recruitable if it is isolated, or if it loses a redundant
pair comparison. A pair is redundant only when both directed edges are
supported and both elapsed-time estimates are at most the configured threshold
`D`. The member with lower total incident supported confidence loses; an exact
tie selects the higher DG index. Isolated vertices are preferred over redundant
losers, followed by lowest incident support. A birth-support value initialized
to one protects each new field until exponential decay takes it to at most
`0.25`.

Policy-buffer HRL uses controllability confidence and `T_ctrl`; all other
configurations use a passive graph formed by consecutive, different, exclusive
behavior-time DG events at most `L` decisions apart in one physical episode.
The passive history is independent of CA3 and crosses rollout boundaries.
Generation IDs reject rollout evidence made stale by reassignment. In HRL the
birth-support clock uses the configured option-event half-life; otherwise it
uses the passive-transition half-life.

For input feature `x` and all other projection rows `W_{-j}`:

$$
w_j\leftarrow\frac{(I-\Pi_{\mathrm{span}(W_{-j})})x}
{\|(I-\Pi_{\mathrm{span}(W_{-j})})x\|}.
$$

The chosen row's optimizer state is cleared and its BatchNorm statistics are
set so the candidate is above threshold by the recruitment margin. The
default cap is one row per accepted rollout. This is a non-gradient
intervention, not an optimizer loss; it invalidates graph information incident
to a replaced landmark in both passive and controllability graphs and advances
their representation generations. `recruitment_committed` remains checkpointed
only as legacy ever-recruited telemetry in graph mode.

### 6.2 Controllability and passive graph fast weights

Graph confidence/arrival statistics update once per newly accepted rollout,
outside autograd. On successful deliberate option `i -> j` in `tau` steps:

$$
\gamma=0.5^{1/h},\qquad C_{ij}\leftarrow\gamma C_{ij}+1,
$$

$$
T^{ctrl}_{ij}\leftarrow
\frac{\gamma C_{ij}^{old}T^{ctrl}_{ij}^{old}+\tau}{C_{ij}}.
$$

These fast weights select future actor targets/routes but cannot change PPO
inputs already stored in a rollout.

## 7. Default And Status Summary

| Mechanism | Default | Important constraints |
|---|---:|---|
| PPO policy/value/entropy | on | Base worker optimization. |
| Temporal-distance reward | on for intrinsic configurations | Encoder method controls sign/centering. |
| HRL hit / hit-distance gating | HRL-specific | Requires active intended target/waypoint. |
| Reward-weighted DG encoder loss | on in intrinsic configurations | PPO gradients to DG are cleared. |
| Batch unused-unit recruitment | batch dependent, usually on | Its magnitude may be negative by construction. |
| Local unused / multi-activation | off | Legacy auxiliary terms. |
| Population usage/density/collision | off | Three coupled coefficients. |
| Global punishment / row repulsion / CA3 exclusion | off | Anti-collapse ablations. |
| Path scatter | off | Requires topological manager + action integration. |
| CA3 predictor | off | Head-only shadow predictor; no manager control. |
| Orthogonal recruitment | off | One non-gradient row replacement per rollout by default. |
