# Minimal CA3 State Readout via Action-Conditioned Innovation Prediction

**Status:** implementation plan  
**Target stack:** IntrMotiv + Sample Factory + PyTorch  
**Primary runtime target:** sf_working_directories/IntrMotiv/ on NEMO2  
**Primary learner path:** ControllerLearner / DistanceLearnerReward

> **Implementation note for downstream agents:** all code in this document is conceptual pseudocode. Do not copy it mechanically or introduce parallel infrastructure just to match the snippets. First inspect the current IntrMotiv/Sample Factory data flow, tensor layouts, replay interfaces, optimizer structure, and naming conventions, then implement the same mathematical mechanism in the smallest way that fits the existing codebase.

## 1. Goal

Learn a compact state/goal representation from the existing DG→CA3 memory without ground-truth coordinates, without adding a second recurrent dynamics model, and without changing DG or CA3 in the first version.

Let the flattened CA3 state be

$$
s_t\in\mathbb R^N,\qquad N=F(R+L-1).
$$

The fixed CA3 update is

$$
s_{t+1}=Js_t+Bu_{t+1},
$$

where J is the fixed CA3 shift operator, B injects the new DG activity into the first R slots, and u is the sparse DG input.

The key decomposition is

$$
\underbrace{Js_t}_{\text{known intrinsic memory dynamics}}
+
\underbrace{Bu_{t+1}}_{\text{new environmental information}}.
$$

Do **not** train another recurrent model to relearn J. Learn only

$$
\boxed{z_t=Ws_t,\qquad z_t\in\mathbb R^{d_z}}
$$

with **raw CA3 activity as the direct input to the readout**.

Do not difference adjacent CA3 slots and do not reconstruct the exact historical DG input sequence before applying W. The repetition factor R already spreads each sparse DG event across R adjacent CA3 positions, giving the CA3 state a fixed temporal smoothing/recency basis. Near the finite-memory boundary, the support of an old event naturally tapers as copies shift out of the register.

This is a useful prior rather than a distortion that must be inverted. The learned readout can:

- sum neighboring CA3 positions when temporal smoothing is useful;
- apply different weights across positions to compensate for the boundary taper;
- use signed differences across positions if sharper event timing is predictive.

Thus the intended computation is directly

$$
\boxed{S_t\rightarrow W\rightarrow z_t}
$$

rather than

$$
S_t\rightarrow\text{recovered DG history}\rightarrow W\rightarrow z_t.
$$

Train W so that z, together with the executed action sequence, predicts the future DG innovations.

For horizon H,

$$
s_{t+H}=J^Hs_t+\sum_{h=1}^{H}J^{H-h}Bu_{t+h}.
$$

Define

$$
\boxed{\eta_{t,H}=s_{t+H}-J^Hs_t}.
$$

V1 should predict the smaller equivalent target

$$
\boxed{U_{t,H}=[u_{t+1},\ldots,u_{t+H}]}
$$

and use J only to reconstruct/check the implied CA3 innovation.

## 2. V1 scope

The first control use-case is **HER**, not a persistent online goal registry.

Implement:

1. linear CA3 readout W;
2. small feed-forward action-conditioned innovation predictor;
3. learner-side self-supervised auxiliary loss from ordinary Sample Factory replay;
4. continual online learning of W from the prediction objective;
5. no auxiliary gradient into DG, visual encoder, or CA3 in v1;
6. no PPO/HER policy gradient into W in v1;
7. HER relabeling that selects a future achieved event by index, retrieves that event's existing replayed CA3 state, and embeds it with the current W;
8. shadow/diagnostic checks before changing the behavior policy.

Do not implement yet:

- another RNN/GRU/LSTM;
- learned CA3 recurrence;
- modification of J;
- coordinate supervision;
- contrastive same-place labels;
- reachability-anchor banks;
- Laplacian/SFA as the primary loss;
- a new persistent goal-memory subsystem;
- automatic goal merging;
- intrinsic reward based on z-distance.

## 3. Initial configuration

Current common setting:

~~~text
F = Hippo_n_feature = 16
R = Hippo_R         = 8
L = Hippo_L         = 64
E = R + L - 1       = 71
N = F * E           = 1136
~~~

Suggested defaults:

~~~text
ca3_state_readout=false
ca3_state_readout_shadow=true
ca3_state_readout_dim=16
ca3_state_readout_horizon=8
ca3_state_readout_hidden=128
ca3_state_readout_aux_coeff=0.1
ca3_state_readout_active_coeff=1.0
ca3_state_readout_zero_coeff=0.1
ca3_state_readout_var_coeff=0.1
ca3_state_readout_cov_coeff=0.01
ca3_state_readout_eta_coeff=0.0
ca3_state_readout_intertwine_coeff=0.0
ca3_state_readout_dg_grad=false
ca3_state_goal_conditioning=false
~~~

H=8 is only a starting value, chosen to match the common R=8 timescale and reduce the dominance of no-event one-step targets.

## 4. PyTorch modules

Keep both modules as submodules of the existing actor-critic so Sample Factory checkpointing and synchronization handle them normally.

~~~python
class CA3StateReadout(nn.Module):
    def __init__(self, ca3_dim, z_dim):
        super().__init__()
        self.linear = nn.Linear(ca3_dim, z_dim, bias=False)

    def forward(self, ca3_flat):
        return self.linear(ca3_flat)


class CA3InnovationPredictor(nn.Module):
    def __init__(self, z_dim, n_actions, horizon, n_dg, hidden=128):
        super().__init__()
        self.horizon = horizon
        self.n_dg = n_dg
        self.net = nn.Sequential(
            nn.Linear(z_dim + horizon * n_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizon * n_dg),
        )

    def forward(self, z_t, action_seq_onehot):
        x = torch.cat(
            [z_t, action_seq_onehot.flatten(start_dim=1)],
            dim=-1,
        )
        return self.net(x).view(-1, self.horizon, self.n_dg)
~~~

Use a strong bottleneck: start with d_z=16; later compare 8/16/32.

## 5. Read raw CA3 directly; use J only on the target/consistency side

The readout input is the raw amplitude-valued CA3 state S itself. No deconvolution or explicit recovery of past DG inputs is part of the representation path.

In the active CA3 implementation, J is a zero-filled shift along the register axis. J is used only to factor known intrinsic CA3 evolution out of future targets or to check innovation consistency.

Represent CA3 as S[..., F, E]:

~~~python
def ca3_shift_power(S, h):
    if h == 0:
        return S
    out = torch.zeros_like(S)
    if h < S.shape[-1]:
        out[..., h:] = S[..., :-h]
    return out
~~~

For target construction only, the current DG injection is available from slot zero:

$$
\boxed{u_t=S_t[...,0]}
$$

because the shifted previous state contributes zero there.

This slot-zero identity is **not** used to reconstruct the past before the readout. It is only a convenient way to obtain future DG supervision from replayed CA3 states without another visual/DG forward.

### What is actually learned

The known CA3 recurrence already predicts the memory-only part of the next state:

$$
J S_t.
$$

The learned model should predict only the new environmental injection:

$$
\boxed{
\hat u_{t+1}=P_\phi(W S_t,a_t).
}
$$

The complete one-step CA3 prediction is then assembled using the known dynamics:

$$
\boxed{
\hat S_{t+1}=J S_t+B\hat u_{t+1}.
}
$$

For the v1 multi-step representation loss, avoid forcing the predictor to recursively relearn CA3 dynamics. Directly predict the sequence of future injections conditioned on the current readout and executed action sequence:

$$
\boxed{
P_\phi(W S_t,a_{t:t+H-1})
=
[\hat u_{t+1},\ldots,\hat u_{t+H}].
}
$$

Train against the actual future DG injections:

$$
\boxed{
\mathcal L_{pred}
=
\sum_{h=1}^{H}
\alpha_h\,
\ell(
\hat u_{t+h},
\operatorname{sg}(u_{t+h})
).
}
$$

The corresponding full CA3 innovation is implied, not separately learned:

$$
\hat\eta_{t,H}
=
\sum_{h=1}^{H}
J^{H-h}B\hat u_{t+h}.
$$

Thus the division of labor is explicit:

~~~text
J                    = known intrinsic CA3 memory evolution
P_phi(W S, actions)   = learned prediction of new incoming DG events
~~~

For diagnostics:

~~~python
eta_true = S_tH - ca3_shift_power(S_t, H)
~~~

Reconstruct innovation from a predicted DG sequence:

~~~python
def inject_dg(u, R, E):
    out = u.new_zeros(*u.shape, E)
    out[..., :R] = u.unsqueeze(-1)
    return out

def reconstruct_innovation(u_seq, R, E):
    B, H, F = u_seq.shape
    eta = u_seq.new_zeros(B, F, E)
    for h in range(H):
        inj = inject_dg(u_seq[:, h], R, E)
        eta = eta + ca3_shift_power(inj, H - 1 - h)
    return eta
~~~

Unit-test the exact identity against repeated execution of the existing CA3 core before enabling any loss.

## 6. Sample Factory data path

Use the existing learner replay forward. Do **not**:

- run the visual encoder/DG a second time;
- assume raw rnn_states are a [stream,time] tensor;
- reconstruct temporal windows from neighboring rows of an already flattened packed sequence.

After the normal replay forward, expose or retain aligned tensors:

~~~text
ca3_seq:          [T, B, F, E]
actions:          [T, B]
valids:           [T, B]
episode_boundary: [T, B]
~~~

ca3_seq[t] must be the CA3 state associated with the observation used to choose actions[t].

Then

~~~python
dg_seq = ca3_seq[..., 0]  # [T,B,F]
~~~

A horizon-H training item is

~~~text
input:    S_t
actions:  a_t ... a_(t+H-1)
targets:  u_(t+1) ... u_(t+H)
future:   S_(t+H)
~~~

The indexing invariant is

$$
S_t\xrightarrow{a_t}u_{t+1},S_{t+1}.
$$

Write a synthetic one-step test for this exact alignment.

If the existing learner forward only exposes flattened core outputs, extend the IntrMotiv-local return object to expose an auxiliary padded CA3 sequence before flattening. Do not patch upstream Sample Factory unless unavoidable.

## 7. Window masks

A start t is valid only if all H transitions are valid and the CA3 state is never reset inside the window.

~~~python
window_valid = (
    valids[t:t+H+1].all(dim=0)
    & ~episode_boundary[t:t+H].any(dim=0)
)
~~~

For this auxiliary, any event that resets CA3 is a boundary, regardless of PPO's bootstrap semantics.

Never learn across an episode/reset boundary.

## 8. Auxiliary forward

~~~python
S0 = ca3_seq[t0, valid]                     # [Bv,F,E]
s0 = S0.flatten(start_dim=1).detach()       # no aux gradient to DG/CA3
z0 = actor_critic.ca3_state_readout(s0)

a_seq = actions[t0:t0+H, valid].T           # [Bv,H]
a_1h = F.one_hot(a_seq, num_classes=n_actions).float()

u_target = (
    ca3_seq[t0+1:t0+H+1, valid, :, 0]
    .permute(1, 0, 2)
    .detach()
)

u_pred = actor_critic.ca3_innovation_predictor(z0, a_1h)
~~~

Gradient policy:

~~~text
W / readout                 YES
innovation predictor        YES
DG                          NO
visual trunk                NO
CA3/J                       NO
policy decoder              NO
manager/reward machinery    NO
~~~

The scientific question is isolated:

> Can a compact readout of the existing CA3 memory predict action-conditioned future landmark innovations?

## 9. Prediction loss

DG is sparse, so do not average all target entries uniformly.

~~~python
err = F.smooth_l1_loss(u_pred, u_target, reduction="none")
active = u_target.abs() > cfg.dg_event_epsilon

l_active = masked_mean(err, active)
l_zero = masked_mean(err, ~active)

l_pred = (
    cfg.ca3_state_readout_active_coeff * l_active
    + cfg.ca3_state_readout_zero_coeff * l_zero
)
~~~

Initial weighting:

~~~text
active_coeff = 1.0
zero_coeff   = 0.1
~~~

If a minibatch has no active targets, define l_active=0 without NaNs and log the event count.

## 10. Non-collapse constraints

Prediction alone can let the predictor ignore z and use average action-conditioned statistics. Add a small variance/covariance constraint.

For centered z, C = z^T z / (B-1):

$$
\mathcal L_{var}
=
\frac1{d_z}\sum_i
[\max(0,1-\sqrt{C_{ii}+\epsilon})]^2
$$

and

$$
\mathcal L_{cov}
=
\frac{1}{d_z(d_z-1)}
\sum_{i\neq j} C_{ij}^2.
$$

~~~python
def variance_covariance_loss(z, eps=1e-4):
    z = z - z.mean(dim=0, keepdim=True)
    cov = z.T @ z / max(z.shape[0] - 1, 1)

    std = torch.sqrt(torch.diagonal(cov) + eps)
    l_var = torch.relu(1.0 - std).pow(2).mean()

    offdiag = cov - torch.diag(torch.diagonal(cov))
    l_cov = offdiag.pow(2).sum() / max(z.shape[1] * (z.shape[1] - 1), 1)
    return l_var, l_cov
~~~

Initial coefficients:

~~~text
var_coeff = 0.1
cov_coeff = 0.01
~~~

These are anti-collapse terms, not the scientific objective.

## 11. J-based innovation consistency

Compute

~~~python
eta_true = (
    ca3_seq[t0+H, valid]
    - ca3_shift_power(ca3_seq[t0, valid], H)
).detach()

eta_pred = reconstruct_innovation(
    u_pred,
    R=cfg.Hippo_R,
    E=cfg.Hippo_R + cfg.Hippo_L - 1,
)

l_eta = F.smooth_l1_loss(eta_pred, eta_true)
~~~

Start with

~~~text
ca3_state_readout_eta_coeff = 0.0
~~~

and log l_eta only.

If DG-sequence prediction looks correct but eta reconstruction does not, treat this as an implementation/alignment bug.

Only after the algebra is verified consider a small eta coefficient such as 0.1.

## 12. Optional intertwining regularizer

Later, optionally test a small latent operator A such that

$$
\boxed{WJ\approx AW}.
$$

Use

$$
\mathcal L_J=\frac1{d_zN}\|WJ-AW\|_F^2.
$$

Do not explicitly construct J; implement WJ via shifts in the CA3 [F,E] layout.

Default coefficient: zero.

This is an ablation, not part of the core v1 objective. A strong exact intertwining constraint may be too restrictive for a low-dimensional quotient of the nonnormal shift-register memory.

## 13. Total loss

V1:

$$
\boxed{
\mathcal L_{state}
=
\mathcal L_{pred}
+\lambda_{var}\mathcal L_{var}
+\lambda_{cov}\mathcal L_{cov}
+\lambda_\eta\mathcal L_\eta.
}
$$

Add

$$
\boxed{
\mathcal L_{total}
=
\mathcal L_{existing}
+\lambda_{state}\mathcal L_{state}.
}
$$

Initial lambda_state=0.1 and lambda_eta=0.

In the current iterative schedule:

- update this module during simultaneous training;
- update during the representation/encoder phase;
- do not update during decoder-only phases.

Keep this as a separately named loss, not hidden inside an existing DG objective.

## 14. Gradient ownership and simultaneous learning

Use three conceptually separate objectives:

$$
\boxed{
\begin{aligned}
\theta_{DG} &\leftarrow \mathcal L_{DG}^{existing},\\
(W,\phi) &\leftarrow \mathcal L_{pred},\\
(\theta_\pi,\theta_V) &\leftarrow \mathcal L_{RL/HER}.
\end{aligned}
}
$$

Here $\phi$ denotes the innovation predictor.

In v1:

$$
\boxed{
\frac{\partial \mathcal L_{pred}}{\partial \theta_{DG}}=0
}
$$

and

$$
\boxed{
\frac{\partial \mathcal L_{RL/HER}}{\partial W}=0.
}
$$

Thus W keeps changing online, but only because it is learning a predictive state representation. The controller continuously adapts to the current goal embedding, but its RL gradient does not deform the goal space yet.

Conceptually:

~~~text
existing DG loss  ----------> DG

prediction loss   ----------> W
                  \---------> innovation predictor

PPO/HER loss      ----------> actor / critic

prediction loss   -X-------> DG
PPO/HER loss      -X-------> W
~~~

Train these objectives **simultaneously on the same learner updates** when practical. There is no need to alternate "train W / freeze W / train controller" phases in the first design. Detach operations or parameter-specific loss routing are sufficient to keep the gradients separate even if the scalar losses are summed before the backward pass.

Only add alternating optimization if simultaneous updates are empirically unstable.

Later ablations may independently enable:

1. PPO/HER gradient into W, asking whether control should shape the state/goal representation;
2. prediction gradient into DG, asking whether DG should learn events that improve predictive state inference.

Do not enable both couplings in the first experiment.

## 15. Likely IntrMotiv integration points

Keep changes local to IntrMotiv.

Likely files:

~~~text
sf_working_directories/IntrMotiv/dmlab/custom_actor_critic.py
sf_working_directories/IntrMotiv/dmlab/custom_learner.py
sf_working_directories/IntrMotiv/dmlab/train_hipposlam.py
~~~

and, only if needed to expose replay CA3:

~~~text
the IntrMotiv file containing BypassSS / the custom core
~~~

Suggested new helper:

~~~text
sf_working_directories/IntrMotiv/dmlab/ca3_state_readout.py
~~~

Do not refactor BypassSS merely to implement this experiment.

## 16. Shadow-mode requirement

First implementation is observational.

With readout enabled but goal conditioning disabled, the following must be unchanged relative to the parent:

- actor logits before the new modules receive updates;
- intrinsic reward;
- DG update;
- CA3 update;
- manager target selection;
- existing FiLM goal input;
- graph updates;
- episode termination;
- behavior-goal replay.

Only W, the predictor, and telemetry should change.

## 17. Logging

At minimum:

~~~text
state_readout/pred_loss
state_readout/pred_active_loss
state_readout/pred_zero_loss
state_readout/eta_loss
state_readout/z_mean_abs
state_readout/z_std_mean
state_readout/z_std_min
state_readout/z_cov_offdiag_rms
state_readout/active_target_fraction
state_readout/valid_windows
state_readout/grad_W
state_readout/grad_predictor
~~~

Also evaluate without gradients:

1. state shuffle: permute z across windows, preserve actions;
2. action shuffle: permute action sequences, preserve z.

Log

~~~text
state_readout/state_shuffle_delta
state_readout/action_shuffle_delta
~~~

where delta = shuffled loss - normal loss.

Desired:

$$
\boxed{
\Delta_{state}>0,\qquad \Delta_{action}>0.
}
$$

If state shuffle does nothing, the predictor is not using the learned state.
If action shuffle does nothing, the action-conditioning is not contributing.

## 18. Offline representation evaluation

Coordinates may be used only for evaluation, not training.

Measure:

- kNN position error from z;
- optional linear probe z→(x,y);
- distance between same-location visits reached through different routes;
- distance between different physical fields of the same DG unit;
- heading separation at matched positions;
- collision rate: physically distant states with small z-distance.

Central desired relation:

$$
\boxed{
d_z(\text{same place, different route})
<
d_z(\text{different field of same DG unit}).
}
$$

Compare this against raw contextual CA3 and current DG identity.

## 19. HER-first goal representation

For the first controller experiment, do **not** build a persistent continuous-goal database.

A HER goal is already identified by a future achieved event inside the replayed trajectory. Let $e$ be the selected future event index. The replay already contains the corresponding CA3 state $S_e$.

Use the index only as a temporary pointer:

$$
\boxed{
e \longrightarrow S_e \longrightarrow g_e=W S_e.
}
$$

For the current state:

$$
z_t=W S_t.
$$

Therefore current state and goal are always embedded by the **same current W**:

$$
\boxed{
z_t,g_e\in\mathcal G=\mathbb R^{d_z}.
}
$$

This requires no new learned memory component and no stored numerical goal embedding.

### Why not store only a time index permanently?

A time/event index is elegant while the referenced state remains available in the current rollout/replay or CA3 history. It is not a persistent goal representation: once the referenced trajectory/state is no longer accessible, the index has no content. In an online implementation tied directly to the finite CA3 register, such a pointer would also expire with the memory horizon; an H-step prediction target further shortens the usable future window.

For **HER first**, this is not a problem: select $e$ within the existing same-episode replay and immediately retrieve $S_e$.

For a later online controller that must pursue a goal after the original CA3 state has shifted out or after the trajectory has left replay, store a detached **raw CA3 snapshot** $\tilde g=S_{t_g}$ rather than a frozen learned vector $g$. Then re-embed it whenever it is used:

$$
\boxed{
g=W\tilde g.
}
$$

This is still only a snapshot of the existing CA3 buffer, not a new recurrent memory mechanism.

## 20. Goal conditioning of the controller

Keep the full current CA3 state available to the controller:

$$
\boxed{
\pi(a_t\mid S_t,\text{depth}_t,\operatorname{sg}(W S_e)).
}
$$

The stop-gradient is with respect to the PPO/HER loss only. W is still updated concurrently by $\mathcal L_{pred}$.

The first continuous-goal FiLM implementation can map the current goal embedding through the existing modulation interface, e.g. conceptually:

~~~python
goal_z = state_readout(goal_ca3)
goal_z_for_policy = goal_z.detach()
delta_gamma_beta = goal_adapter(goal_z_for_policy)
~~~

This snippet is illustrative only. Reuse the existing goal replay / FiLM pathway and its tensor conventions rather than creating a parallel controller path.

Do **not** replace the controller's current full CA3 input by $z_t$ in v1. The full CA3 state may contain orientation, recent-event timing, and other control-relevant information that the abstract goal space intentionally discards.

## 21. HER training with a moving W

Reuse the existing empirical same-episode future-goal machinery.

For each HER source segment:

1. select a future achieved qualified event index $e$ using the existing replay alignment;
2. retrieve its replayed raw CA3 state $S_e$;
3. compute the goal with the **current** readout, $g_e=W S_e$;
4. stop the PPO/HER gradient at $g_e$;
5. keep that goal condition fixed across the relabeled source segment for that learner evaluation;
6. keep the existing empirical terminal pseudo-return / achieved-event semantics initially.

There is no requirement to freeze W between learner updates. If W changes from one update to the next, both current states and future-goal states are re-embedded by the same current mapping.

Do not store $W_{old}S_e$ as the canonical goal. Store/recover $S_e$ and compute $W S_e$ on demand.

For this first HER experiment, the selected achieved event $e$ itself defines success. Do **not** yet introduce a threshold such as $\|W S_t-W S_e\|<\epsilon$ as the reward criterion. That would make changing W also change reward semantics and would confound the representation experiment.

## 22. Minimal ablations

### Representation preflight

Same seed:

~~~text
S0: parent, no readout
S1: shadow readout + action-conditioned innovation
S2: same readout but action information zeroed/shuffled
~~~

Questions:

- does S1 preserve parent behavior initially?
- does prediction use both state and action?
- does S1 produce better place disambiguation and route invariance than raw CA3?

### HER goal representation comparison

After the readout has passed basic prediction/alignment checks, compare:

~~~text
A: current DG one-hot HER goal
B: raw contextual CA3 HER goal through the existing goal adapter
C: learned HER goal g = W S_e, recomputed from the selected future replay state
~~~

Keep DG, CA3, worker architecture, optimizer budget, HER settings, reward scale, and seeds matched.

This directly tests:

~~~text
DG identity
vs.
raw contextual identity
vs.
learned predictive state abstraction
~~~

## 23. Required regression tests

### CA3 algebra

1. one-step ca3_shift_power equals the existing zero-injection core shift;
2. H-step shift equals H repeated one-step shifts;
3. synthetic DG sequence satisfies
   s_(t+H)-J^H s_t = sum J^(H-h) B u_(t+h);
4. CA3 slot 0 equals the active implementation's current DG injection.

### Alignment

5. action[t] predicts DG[t+1], not DG[t];
6. no window crosses a CA3 reset;
7. no window crosses packed replay streams;
8. invalid/stale Sample Factory entries never enter the auxiliary.

### Gradients

9. W gets nonzero auxiliary gradient;
10. predictor gets nonzero auxiliary gradient;
11. DG gets exactly zero auxiliary gradient;
12. frozen visual trunk gets zero auxiliary gradient;
13. policy decoder gets zero auxiliary gradient in shadow mode.

### Behavior preservation

14. shadow mode preserves actor logits before readout training;
15. shadow mode preserves intrinsic reward;
16. shadow mode preserves manager targets;
17. checkpoint round-trip reproduces W and predictor outputs.

### HER goal mode

18. continuous FiLM zero initialization is exact identity;
19. HER stores/selects the future event index and retrieves the correct replayed CA3 state;
20. the same current W is used for both current-state and future-goal embedding;
21. PPO/HER loss gives zero gradient to W while prediction loss gives nonzero gradient to W;
22. HER relabel uses a future achieved event from the same episode and keeps its goal condition fixed across the relabeled segment;
23. no persistent continuous-goal database is required for the HER-first path.

Run focused tests first, then the authoritative full IntrMotiv suite on NEMO2.

## 24. Suggested code organization

~~~text
dmlab/
    ca3_state_readout.py
    custom_actor_critic.py
    custom_learner.py
    train_hipposlam.py
    tests/
        test_ca3_state_readout.py
        test_ca3_state_alignment.py
        test_ca3_state_gradients.py
        test_ca3_state_goal_replay.py
~~~

## 25. Learner-side pseudocode

~~~python
def ca3_state_readout_loss(actor_critic, ca3_seq, actions, valids,
                           episode_boundary, cfg):
    H = cfg.ca3_state_readout_horizon
    A = actor_critic.action_space.n

    zs, preds, targets = [], [], []
    eta_preds, eta_trues = [], []

    T = ca3_seq.shape[0]

    for t in range(T - H):
        valid = valids[t:t+H+1].all(dim=0)
        valid &= ~episode_boundary[t:t+H].any(dim=0)

        if not valid.any():
            continue

        S0 = ca3_seq[t, valid]
        SH = ca3_seq[t+H, valid]

        target_u = (
            ca3_seq[t+1:t+H+1, valid, :, 0]
            .permute(1, 0, 2)
            .detach()
        )

        action_seq = actions[t:t+H, valid].permute(1, 0)
        action_1h = F.one_hot(action_seq, A).float()

        z = actor_critic.ca3_state_readout(
            S0.flatten(start_dim=1).detach()
        )

        pred_u = actor_critic.ca3_innovation_predictor(z, action_1h)

        eta_true = (
            SH - ca3_shift_power(S0, H)
        ).detach()

        eta_pred = reconstruct_innovation(
            pred_u,
            R=cfg.Hippo_R,
            E=cfg.Hippo_R + cfg.Hippo_L - 1,
        )

        zs.append(z)
        preds.append(pred_u)
        targets.append(target_u)
        eta_preds.append(eta_pred)
        eta_trues.append(eta_true)

    if not zs:
        return zero_loss_dict()

    z = torch.cat(zs, 0)
    pred = torch.cat(preds, 0)
    target = torch.cat(targets, 0)
    eta_pred = torch.cat(eta_preds, 0)
    eta_true = torch.cat(eta_trues, 0)

    err = F.smooth_l1_loss(pred, target, reduction="none")
    active = target.abs() > cfg.dg_event_epsilon

    l_active = masked_mean(err, active)
    l_zero = masked_mean(err, ~active)

    l_pred = (
        cfg.ca3_state_readout_active_coeff * l_active
        + cfg.ca3_state_readout_zero_coeff * l_zero
    )

    l_var, l_cov = variance_covariance_loss(z)
    l_eta = F.smooth_l1_loss(eta_pred, eta_true)

    l_state = (
        l_pred
        + cfg.ca3_state_readout_var_coeff * l_var
        + cfg.ca3_state_readout_cov_coeff * l_cov
        + cfg.ca3_state_readout_eta_coeff * l_eta
    )

    return cfg.ca3_state_readout_aux_coeff * l_state
~~~

Vectorize only after the indexing/algebra tests pass.

## 26. Success gate before using z as a goal

Do not proceed merely because training loss decreases.

Require:

1. state shuffle increases prediction loss;
2. action shuffle increases prediction loss;
3. z does not collapse;
4. different physical fields of one DG unit separate better than under DG identity;
5. same-place revisits through different routes become closer than in raw contextual CA3;
6. auxiliary gradient into DG/visual/CA3 is exactly zero;
7. shadow mode preserves current controller/reward behavior.

If these fail, stop rather than adding more structure.

## 27. Interpretation

The intended mechanism is

~~~text
visual input
    ↓
sparse DG event u_t
    ↓
fixed CA3 sequence memory S_t
    ↓
R-wide temporal smoothing / finite-memory recency basis
    ↓
direct linear bottleneck z_t = W S_t
    ↓
predict action-conditioned future DG innovations
~~~

The CA3 basis is consumed directly. There is no explicit inversion back to the exact input history. R supplies a useful prior toward temporal smoothness and recency, while W is free to preserve that smoothing, compensate for tapering near the horizon, or recover sharper timing through signed position-dependent weights when useful.

J is not relearned; it is used only to factor known intrinsic CA3 evolution out of the future-side prediction/consistency target.

This is the self-supervised analogue of a ground-truth-supervised CA3→state association:

$$
S_t\xrightarrow{W} s_t^{true}
$$

becomes

$$
\boxed{
S_t\xrightarrow{W}z_t
\quad\text{such that}\quad
(z_t,a_{t:t+H-1})
\text{ predicts future environmental innovations}.
}
$$

Desired behavior:

~~~text
irrelevant route-history differences
    -> discarded by the bottleneck

history information required to resolve DG aliasing
    -> retained

action-relevant latent state
    -> retained
~~~

If this works, the same z is the natural candidate goal space:

$$
\boxed{g=z_{t_g}}.
$$

## 28. Execution order

1. implement CA3 shift/injection algebra helpers and tests;
2. add W and the feed-forward innovation predictor;
3. expose action-aligned padded replay CA3;
4. implement H-step windows and boundary masks;
5. implement prediction + variance/covariance loss;
6. verify that prediction updates W but not DG, and PPO/HER updates the controller but not W;
7. run one-seed shadow preflight;
8. add state/action shuffle diagnostics;
9. integrate **same-episode HER first**: select future event index e, retrieve S_e, compute g=W S_e, detach g for the policy loss;
10. verify exact replay alignment and fixed goal conditioning across each relabeled segment;
11. run offline coordinate-based representation diagnostics;
12. compare action-conditioned versus action-ablated prediction;
13. compare DG-one-hot, raw-CA3, and learned-W HER goal representations;
14. only after the HER result is understood consider persistent online goals, storing raw CA3 snapshots if necessary.

## 29. Main decision rule

If the action-conditioned readout does **not** outperform raw contextual CA3 on

$$
\text{cross-route same-place consistency}
$$

and

$$
\text{separation of different fields of the same DG unit},
$$

stop. Do not add successor features, anchor banks, learned goal merging, or a larger world model.

If it does, use $z=Ws$ as the default candidate state/goal representation for the HER-first IntrMotiv controller experiment. Persistent online goal storage is a later question.
