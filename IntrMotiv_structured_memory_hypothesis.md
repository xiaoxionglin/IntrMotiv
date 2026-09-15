# IntrMotiv: Structured Memory as a Substitute for Rich Instantaneous Encoding

## Core intuition

The DG→CA3 architecture may be understood as trading **instantaneous sensory richness** for **structured temporal memory**:

\[
\boxed{\text{spend less computation on each observation; spend more state capacity on structured history}}
\]

Rather than constructing a maximally informative representation from every frame, the agent can use a sparse/cheap DG code for informative sensory events and define the controller's effective state as a structured temporal expansion of those events.

A conventional recurrent agent roughly does

\[
o_t \xrightarrow{\text{encoder}} z_t,\qquad
h_t=f_\theta(h_{t-1},z_t),
\]

so it must repeatedly learn what to encode, retain, forget, and update.

Our architecture instead uses

\[
o_t \xrightarrow{\text{DG}} u_t,
\]

followed by fixed CA3 dynamics

\[
\boxed{x_{t+1}=Sx_t+Ju_t}.
\]

When there is no informative new DG input,

\[
u_t=0,
\]

the memory evolves as

\[
\boxed{x_{t+\tau}=S^\tau x_t}.
\]

Thus the instantaneous DG representation can be relatively poor because the state available to the controller is the accumulated, structured temporal representation \(x_t\).

## CA3 is not merely memory: it imposes a trajectory prior

The fixed CA3 shift dynamics provide a strong inductive bias.

After an event has entered CA3, its subsequent evolution is known in advance until new evidence arrives. The architecture therefore assumes:

> Once an informative event occurs, its internal trace follows a fixed temporal trajectory unless another input changes the state.

This differs from an LSTM or generic RNN, which must learn at every step:
- what to retain,
- what to forget,
- how elapsed time changes the representation,
- how sensory input modifies memory.

CA3 hard-codes much of this evolution. A high-dimensional temporal state can therefore be maintained with very cheap computation: mainly shifting/permuting an existing sparse state and inserting new activity.

This suggests an alternative design philosophy:

| Architecture | Sensory representation | Temporal representation |
|---|---|---|
| Feedforward CNN | rich | none |
| LSTM/RNN | rich/moderate | learned/compressed |
| Transformer | rich | large/flexible |
| DG→CA3 | sparse | large but highly structured |

The hypothesis is therefore not simply that memory is useful, but that **structured memory can substitute for instantaneous representational richness**.

## Event-driven rather than frame-driven state construction

Navigation observations are often highly redundant:

\[
o_t \approx o_{t+1} \approx o_{t+2}.
\]

It may be wasteful to reconstruct a rich semantic state from every frame.

Instead, the system can operate around informative events:

\[
E_A \longrightarrow \longrightarrow \longrightarrow E_B \longrightarrow \longrightarrow E_C.
\]

DG approximately determines when an event is worth registering. CA3 then represents how long ago that event occurred and propagates its trace through a predetermined temporal trajectory.

This creates an **event-driven state representation**:
- sparse sensory updates when something informative happens;
- cheap deterministic memory evolution in between.

## Policy may only need to matter at branch points

A further implication is that much of internal state evolution can be **policy-independent**.

Between informative inputs,

\[
x_t \rightarrow Sx_t \rightarrow S^2x_t \rightarrow \cdots
\]

requires no action-conditioned inference.

Policy becomes important when the world offers meaningful alternatives—for example at a junction, branch point, or newly detected event:

\[
\text{default trajectory}
\quad\rightarrow\quad
\begin{cases}
\text{trajectory A}\\
\text{trajectory B}\\
\text{trajectory C}
\end{cases}
\]

This suggests that instead of learning a fully action-conditioned model at every timestep,

\[
p(s_{t+1}\mid s_t,a_t),
\]

the architecture can treat large parts of experience as predictable trajectory segments and allocate control capacity primarily where trajectories branch.

The computational hypothesis is:

\[
\boxed{\text{many navigation tasks contain long predictable segments separated by sparse decision points}}
\]

and CA3 exploits this structure directly.

## Intrinsic motivation as resource allocation

This framing also gives intrinsic motivation a more specific role.

Novelty need not mean simply that a pixel observation is unusual. It can instead mean:

\[
\boxed{\text{incoming sensory evidence is poorly explained by currently active CA3 trajectories}}
\]

Such a mismatch can indicate:
- a genuinely new landmark,
- an unexpected transition,
- a branch point,
- a changed environment,
- or a failure of the current temporal model.

The system could therefore allocate more representational and exploratory resources only when needed:

\[
\text{predictable trajectory} \rightarrow \text{cheap propagation}
\]

versus

\[
\text{prediction failure / novel event} \rightarrow \text{recruit DG representation + learn transition + explore}.
\]

Intrinsic motivation then becomes a mechanism for **adaptive computational resource allocation**, not generic curiosity.

## Key empirical prediction: vary decision density

This inductive bias should help only when the environment actually contains predictable temporal structure.

A clean test is to vary **decision density** while holding other factors as constant as possible.

Low decision density:

\[
\text{corridor}
\rightarrow
\text{corridor}
\rightarrow
\text{corridor}
\rightarrow
\boxed{\text{junction}}
\rightarrow
\text{corridor}.
\]

High decision density:

\[
\boxed{\text{choice}}
\rightarrow
\boxed{\text{choice}}
\rightarrow
\boxed{\text{choice}}
\rightarrow
\boxed{\text{choice}}.
\]

Prediction:
- DG→CA3 should benefit strongly when informative events/branch points are sparse.
- Its advantage should shrink as optimal control depends on action at every timestep.
- Generic recurrent models may become relatively stronger in the high-decision-density regime because the fixed trajectory prior becomes less appropriate.

This directly tests whether the fixed CA3 dynamics provide the intended inductive bias.

## Sharpened hypothesis

A concise formulation is:

\[
\boxed{
\begin{aligned}
&\text{Use sparse sensory events rather than richly encoding every observation;}\\
&\text{expand those events into a high-dimensional but computationally cheap temporal state;}\\
&\text{hard-code predictable evolution between events;}\\
&\text{spend learning and control capacity primarily where trajectories branch or predictions fail.}
\end{aligned}}
\]

This gives the fixed DG→CA3 architecture a computational justification beyond biological plausibility.

The core claim becomes:

> **Structured memory can substitute for instantaneous representational richness when the environment contains predictable temporal structure.**

This also fits the existing observation that the CA3 agent is particularly effective under sparse DG input while LSTM baselines struggle: the architecture may succeed not by learning a richer instantaneous state, but by using a strong temporal prior to turn sparse sensory events into a useful dynamical state.
