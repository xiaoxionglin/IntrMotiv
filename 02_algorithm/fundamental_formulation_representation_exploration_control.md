# Fundamental formulation: representation, exploration, control, and DG–CA3 memory

Status: working theory note, 2026-09-18.  
Scope: conceptual formulation; not an implementation contract.

The project can be framed as three coupled problems on an unknown latent manifold that is observed through high-dimensional sensory input and can only be sampled by local actions:

1. **Representation:** construct a useful code of the manifold.
2. **Exploration:** generate trajectories that sufficiently cover the manifold.
3. **Control:** use the code, possibly together with memory, to reach specified goals.

The DG–CA3 architecture is one candidate solution: sparse DG events provide a coarse representation, and CA3 supplies structured temporal memory that can resolve ambiguities left by the instantaneous DG code.

---

## 1. Latent manifold and sensory embedding

Let the latent state be

$$
q=(x,y,r)\in\mathcal M,
$$

where \(x,y\) denote physical position and \(r\) is the third pose variable in the current working formulation (e.g. orientation / heading).

The agent does not directly observe \(q\). Instead it receives a high-dimensional sensory observation or feature vector

$$
z=F(q)\in\mathbb R^D.
$$

Thus the realizable observations occupy a low-dimensional manifold embedded in a high-dimensional sensory space:

$$
F(\mathcal M)\subset\mathbb R^D.
$$

The agent does **not** have access to i.i.d. or uniform samples from \(\mathcal M\). Samples arise sequentially from locally constrained dynamics,

$$
q_{t+1}\sim P(\cdot\mid q_t,a_t).
$$

Therefore the state distribution is itself controlled by the policy.

---

## 2. Problem I: representation

Let

$$
h:\mathcal M\rightarrow\mathcal Z
$$

be the learned representation.

The representation need not be one-hot, landmark-like, or metric preserving.

### 2.1 Landmark representation as a special case

For a discrete landmark set

$$
L=\{\ell_1,\ldots,\ell_K\}\subset\mathcal M,
$$

define the covering radius

$$
R_{\rm landmark}
=
\sup_{q\in\mathcal M}
\min_{\ell_i\in L}
d_{\mathcal M}(q,\ell_i).
$$

Small \(R_{\rm landmark}\) means that every point on the manifold lies close to at least one landmark. A useful landmark system should also avoid excessive redundancy, so that many landmarks do not collapse onto nearly the same patch.

This is the clean formulation for a discrete DG-landmark interpretation.

### 2.2 General distributed representation

For a non-one-hot population code, a strong geometric criterion is whether the latent state can be reconstructed up to some resolution:

$$
R_{\rm geom}(h)
=
\inf_D
\sup_{q\in\mathcal M}
d_{\mathcal M}\!\left(q,D(h(q))\right),
$$

where \(D:\mathcal Z\rightarrow\mathcal M\) is an ideal decoder.

However, exact or near-exact geometric reconstruction is stronger than necessary for behavior. The representation may intentionally alias different latent states if the ambiguity has little cost for control or can be resolved from temporal context.

Therefore the more general criterion is:

> The representation, together with the available memory, should retain enough information for good control at the task-relevant resolution.

This makes the landmark-cover objective a special case rather than the definition of representation quality.

---

## 3. Problem II: exploration / sampling coverage

Let the set of states visited by time \(T\) be

$$
V_T=\{q_0,\ldots,q_T\}.
$$

Define the visitation covering radius

$$
R_{\rm visit}(T)
=
\sup_{q\in\mathcal M}
\min_{t\le T}
d_{\mathcal M}(q,q_t).
$$

Then \(R_{\rm visit}(T)\le\epsilon\) means that the trajectory forms an \(\epsilon\)-cover of the manifold.

This is deliberately weaker than requiring uniform visitation. The project does not require every region to be visited equally often; it requires that no large region remain unsampled.

The key distinction is:

- **representation coverage:** how well experienced states are encoded;
- **exploration coverage:** how well experience itself covers \(\mathcal M\).

Improving the representation alone cannot recover a region that is never visited.

For a landmark representation of the visited set, one can define

$$
R_{\rm DG\mid V_T}
=
\sup_{v\in V_T}
\min_{\ell_i\in L}
d_{\mathcal M}(v,\ell_i).
$$

Then, by the triangle inequality,

$$
R_{\rm landmark}
\le
R_{\rm visit}
+
R_{\rm DG\mid V_T}.
$$

This decomposition is useful for separating two failure modes:

- poor exploration: large \(R_{\rm visit}\);
- poor representation of experience: large \(R_{\rm DG\mid V_T}\).

---

## 4. Problem III: control

Let the policy receive an internal control state \(s_t\) and a goal representation \(g\):

$$
\pi(a_t\mid s_t,g).
$$

If the instantaneous representation \(h(q_t)\) uniquely identifies the latent state, then there is no perceptual aliasing and control reduces to ordinary RL: learn which actions in the represented state lead to higher expected future return,

$$
Q(h(q_t),a_t;g)
=
\mathbb E\!\left[
\sum_{k\ge0}\gamma^k r_{t+k}
\right].
$$

Unique coding therefore makes control straightforward to formulate, but it is not necessary.

### 4.1 Representation-induced control loss

Let \(\Pi_{h,L}\) be the policy class allowed to use \(h(q_t)\) together with at most \(L\) steps of history.

Define

$$
V^*_{h,L}(q,g)
=
\max_{\pi\in\Pi_{h,L}}V^\pi(q,g),
$$

and the value loss caused by the representation-memory constraint

$$
\Delta_{\rm ctrl}(h,L)
=
\sup_{q,g}
\left[
V^*(q,g)-V^*_{h,L}(q,g)
\right].
$$

The representation is adequate for control when \(\Delta_{\rm ctrl}(h,L)\) is small.

This permits **recoverable aliasing**. Two latent states may have the same instantaneous representation,

$$
h(q)=h(q'),
$$

even if the optimal immediate actions differ. This is acceptable when:

1. the forced shared action incurs only limited regret;
2. subsequent information separates the trajectories quickly enough;
3. the policy can then recover and make good later decisions.

Thus the important question is not whether aliasing exists, but how much control performance is lost because of it.

---

## 5. Two distinct ways aliasing can be resolved

It is important to distinguish **past-history disambiguation** from **future probing**.

### 5.1 Past-history disambiguation

Let \(u_t=h(q_t)\) denote the instantaneous representation and define the information already available before choosing \(a_t\),

$$
\mathcal H_t^-
=
(u_{t-L+1:t},a_{t-L+1:t-1}).
$$

Two currently aliased latent states may satisfy

$$
h(q)=h(q'),
$$

while

$$
\mathcal H_t^-(q)\neq\mathcal H_t^-(q').
$$

Then the controller can distinguish the states **before choosing the current action**.

This is the regime most directly suited to a memory architecture such as DG–CA3.

### 5.2 Future disambiguation / active probing

For a future action sequence

$$
\alpha=(a_t,\ldots,a_{t+k-1}),
$$

define the future observation signature

$$
\mathcal H_{t,k}^+(q;\alpha)
=
(u_t,u_{t+1},\ldots,u_{t+k}).
$$

It may be that

$$
h(q)=h(q'),
$$

but

$$
\mathcal H_{t,k}^+(q;\alpha)
\neq
\mathcal H_{t,k}^+(q';\alpha).
$$

Here the ambiguity is only resolved **after acting**. The agent may need to take an information-gathering action before it knows which latent state it occupied.

Past-history disambiguation and future probing are related observability questions, but they are not equivalent for online control.

A particularly useful benchmark regime for DG–CA3 is therefore

$$
h(q_t)\ \text{ambiguous},
\qquad
\mathcal H_t^-\ \text{disambiguating}.
$$

---

## 6. Memory advantage as a task property

The current architecture should be evaluated on tasks for which memory is genuinely useful rather than merely available.

Define the memory advantage

$$
A_{\rm memory}(L)
=
\Delta_{\rm ctrl}(h,0)
-
\Delta_{\rm ctrl}(h,L).
$$

A task is especially well matched to the architecture when

$$
\Delta_{\rm ctrl}(h,0)\gg 0,
$$

but

$$
\Delta_{\rm ctrl}(h,L)\approx 0.
$$

In words:

> the instantaneous sparse representation is insufficient, but recent sparse-event history makes good control possible.

This is a stronger justification for CA3 than simply showing that the network contains a recurrent memory.

---

## 7. Goal representation is a separate object

The code used to represent the current state need not be identical to the code used to specify a goal.

Let

$$
\gamma:\mathcal M\rightarrow\mathcal G
$$

be the goal map, with

$$
g=\gamma(q_{\rm target}).
$$

A goal corresponds to a target set

$$
G_g
=
\{q\in\mathcal M:\gamma(q)\approx g\}.
$$

For example, the state representation may need to distinguish \((x,y,r)\), while a goal may intentionally ignore orientation and depend only on \((x,y)\).

This separation is important because using \(h(q)\) directly as a goal can be badly defined when the representation is broad, overlapping, or behaviorally ambiguous.

A generic roaming policy can then achieve substantial reward without meaningfully conditioning on the requested goal.

A simple diagnostic is the goal-conditioning gap

$$
\Delta_{\rm goal}
=
P(\text{success}\mid\text{correct goal})
-
P(\text{success}\mid\text{shuffled goal}).
$$

If

$$
\Delta_{\rm goal}\approx 0,
$$

the policy is effectively ignoring the goal.

---

## 8. Current DG–CA3 formulation

Let DG provide a sparse population code

$$
u_t=h(q_t).
$$

The simplest CA3 model is a shift-register recurrence

$$
x_{t+1}=Sx_t+Ju_t.
$$

Here:

- \(u_t\) is the current sparse DG activity;
- \(J\) injects current DG activity into the beginning of the CA3 sequence;
- \(S\) deterministically advances previously injected activity through the sequence.

A DG event at time \(t_0\) therefore contributes approximately

$$
S^\tau J u_{t_0}
$$

to the CA3 state \(\tau\) steps later.

The basic interpretation is

$$
\text{CA3 state}
\approx
(\text{which DG events occurred},\ \text{how long ago}).
$$

For a sufficiently sparse DG representation, this produces an explicit memory of recent landmark identity and elapsed time.

---

## 9. Why elapsed-time memory can resolve spatial aliasing

Suppose two current states have similar or identical instantaneous DG representations,

$$
u(q)=u(q').
$$

They may still be distinguishable if one state was reached, for example, seven steps after landmark \(i\), while the other was reached eighteen steps after landmark \(j\).

Then the CA3 state contains information approximately of the form

$$
(\text{recent landmark identity},\ \text{elapsed time since landmark}),
$$

which can disambiguate the current latent state even when \(u_t\) alone cannot.

This motivates environments with:

- perceptually aliased corridors;
- repeated rooms or motifs;
- junctions whose correct action depends on a previous landmark;
- merged routes where future observations no longer reveal which route was taken.

The last case is especially diagnostic because past memory is useful while future probing may be impossible.

---

## 10. Action-conditioned CA3

Elapsed time alone is insufficient when the same landmark can lead through different action sequences to different latent states at the same elapsed time.

A first extension is to make sequence progression action dependent,

$$
x_{t+1}
=
S^{\Delta(a_t)}x_t
+
Ju_t.
$$

This can make CA3 track effective travelled distance rather than wall-clock time, but it generally does **not** preserve action order: many action sequences can produce the same total exponent.

A more expressive formulation is

$$
x_{t+1}=S_{a_t}x_t+Ju_t.
$$

### 10.1 Important architectural constraint: preserve DG landmark identity

If action matrices directly permute the DG identity coordinates, then taking an action can make activity originating from landmark \(i\) look like activity from landmark \(j\). That destroys the clean interpretation of DG identity.

A better factorization separates:

1. sequence age;
2. landmark identity;
3. route / action context.

Let the CA3 event state factor as

$$
\text{age}\otimes\text{landmark}\otimes\text{route context}.
$$

Then use

$$
\boxed{
S_a
=
S_{\rm age}
\otimes
I_F
\otimes
R_a
}
$$

where:

- \(S_{\rm age}\) advances the temporal sequence;
- \(I_F\) leaves DG landmark identity unchanged;
- \(R_a\) transforms a small route-context state according to action \(a\).

The injection can be written as

$$
J
=
e_0\otimes I_F\otimes c_0,
$$

where \(e_0\) inserts at sequence age zero and \(c_0\) is the initial route-context vector.

If landmark \(j\) fires at \(t_0\), then after actions
\(a_{t_0},\ldots,a_{t_0+k-1}\), its contribution becomes

$$
e_k
\otimes
e_j
\otimes
\left(
R_{a_{t_0+k-1}}
\cdots
R_{a_{t_0}}
c_0
\right).
$$

Thus CA3 can represent approximately

$$
(\text{landmark identity},\ \text{age},\ \text{route signature since landmark}).
$$

The landmark identity remains explicit while action history modifies only the route-context factor.

To make action order matter, the transforms should generally be non-commuting:

$$
R_aR_b\neq R_bR_a.
$$

Then left-then-right and right-then-left can lead to different internal route signatures.

This is a candidate extension, not yet a commitment to the implementation.

---

## 11. Alternative: store action efference copy explicitly

A simpler control is to leave \(S\) fixed and inject actions into the same memory:

$$
x_{t+1}
=
Sx_t
+
J_u u_t
+
J_a e(a_t).
$$

This explicitly stores recent DG events and recent actions.

It is less structurally elegant than an action-modulated recurrent transition, but it provides a clean baseline for testing whether action history itself is the missing information.

A useful progression of models is therefore:

1. **DG only:** current sparse representation;
2. **DG + fixed-shift CA3:** landmark identity + elapsed time;
3. **DG + CA3 + explicit action memory:** exact recent action evidence;
4. **DG + action-modulated CA3:** route-dependent internal dynamics.

---

## 12. Current high-level hypothesis

The project should not require DG to construct a lossless spatial map.

A more distinctive hypothesis is:

> DG should produce a sparse coarse representation whose ambiguities can be resolved, when necessary, by structured CA3 memory; intrinsic motivation should drive locally constrained trajectories so that this representation-memory system progressively covers the reachable manifold; and the controller should use the resulting state/history representation to reach explicitly defined goals.

This makes the three central questions:

### Representation
How much of \(\mathcal M\) can be compressed into sparse DG activity without creating unacceptable control loss?

### Exploration
Can intrinsic motivation drive the locally constrained trajectory so that the visited set becomes a coarse cover of \(\mathcal M\)?

### Control
For the ambiguities left by the instantaneous DG code, when does CA3 history reduce

$$
\Delta_{\rm ctrl}(h,L)
$$

enough to make goal-directed behavior reliable?

The strongest task regime for the current architecture is one in which instantaneous observation is intentionally insufficient, while sparse event history — landmark identity, elapsed time, and possibly route/action context — is sufficient for good control.
