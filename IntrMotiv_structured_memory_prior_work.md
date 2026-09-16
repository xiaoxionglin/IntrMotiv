# IntrMotiv: Prior Work on Sparse Sensory Encoding + Structured Temporal Memory

## Core idea

The working hypothesis is:

\[
\boxed{\text{spend less computation on each observation; spend more state capacity on structured history}}
\]

Instead of constructing a rich instantaneous representation at every timestep, use sparse or event-driven sensory encoding and let a structured temporal memory maintain the state over time.

The current DG→CA3 mechanism can be written as

\[
x_{t+1}=Sx_t+Ju_t,
\]

where \(u_t\) is sparse DG input and \(S\) is a fixed shift/sequence operator.

If there is no new informative input,

\[
u_t=0,
\]

then

\[
x_{t+\tau}=S^\tau x_t.
\]

The central architectural claim is therefore:

> **Replace repeated inference of temporal state with deterministic temporal expansion of sparse sensory events.**

Equivalently:

\[
\boxed{
\text{learn WHAT happened}
+
\text{hard-code HOW memory evolves with time}
}
\]

rather than

\[
\boxed{
\text{learn WHAT happened}
+
\text{learn WHAT to remember}
+
\text{learn HOW memory evolves}.
}
\]

## 1. Predictive State Representations (PSRs)

PSRs reject the assumption that state must be a rich instantaneous latent variable inferred directly from the current observation. Instead, state is represented as a sufficient statistic of history through predictions of future action-observation sequences:

\[
h_t=(o_1,a_1,\ldots,o_t)\rightarrow z_t.
\]

This is conceptually close to the idea that the instantaneous sensory representation itself does not need to constitute the full state.

**Difference:** PSRs primarily perform predictive compression of history, whereas DG→CA3 performs something closer to an explicit temporal expansion of sparse past events.

**Reference:** Littman, Sutton, Singh — *Predictive Representations of State*  
https://papers.neurips.cc/paper/1983-predictive-representations-of-state.pdf

## 2. Reservoir Computing / Echo State Networks

Reservoir computing uses fixed recurrent dynamics,

\[
x_{t+1}=f(W_{\mathrm{res}}x_t+W_{\mathrm{in}}u_t),
\]

while typically training only a downstream readout.

The philosophy is:

\[
\boxed{\text{fixed rich temporal expansion}+\text{cheap learned readout}}
\]

This is mechanically close to

\[
x_{t+1}=Sx_t+Ju_t.
\]

CA3 can be interpreted as a highly structured reservoir.

**Difference:** standard reservoirs usually use generic/random recurrent dynamics, while \(S\) implements a specific delay-line/sequence prior.

\[
\begin{array}{ll}
\text{reservoir computing:}&\text{generic fixed temporal expansion}\\
\text{DG→CA3:}&\text{sparse event-driven, explicitly sequence-structured expansion}
\end{array}
\]

**Reference:** reservoir-computing review  
https://www.mdpi.com/2673-2688/7/2/70

## 3. Phased LSTM

Phased LSTM introduces a time gate so recurrent units update only during selected temporal phases rather than every timestep.

The broad principle is:

\[
\boxed{\text{not every sensory timestep deserves an expensive recurrent update}.}
\]

This is closely related to sparse DG events.

**Difference:** Phased LSTM mainly controls *when* recurrent computation occurs. DG→CA3 additionally imposes deterministic temporal evolution between informative events.

**Reference:** Neil, Pfeiffer, Liu — *Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences*  
https://arxiv.org/abs/1610.09513  
https://papers.neurips.cc/paper_files/paper/2016/hash/5bce843dd76db8c939d5323dd3e54ec9-Abstract.html

## 4. Skip RNN

Skip RNN learns whether the recurrent hidden state should be updated at a given timestep:

\[
o_t\rightarrow\boxed{\text{is this worth updating memory for?}}\rightarrow h_t.
\]

This is very close to the proposed role of DG: only some sensory inputs are important enough to update memory.

The critical difference is what happens when an update is skipped.

Skip RNN typically has

\[
h_{t+1}=h_t,
\]

whereas CA3 has

\[
\boxed{x_{t+1}=Sx_t}.
\]

So:

- Skip RNN: memory is frozen.
- DG→CA3: memory **advances deterministically in time**.

DG→CA3 does not say “nothing happened.” It says:

> Nothing informative happened, therefore all existing event traces should move one temporal step forward.

**Reference:** Campos et al. — *Skip RNN: Learning to Skip State Updates in Recurrent Neural Networks*  
https://research.google/pubs/skip-rnn-learning-to-skip-state-updates-in-recurrent-neural-networks-2/

## 5. Clockwork RNN

Clockwork RNN partitions recurrent hidden state into modules operating at different temporal frequencies. Some modules update rapidly, while others update slowly.

This introduces temporal structure by design rather than asking generic recurrence to learn every timescale.

**Difference:** Clockwork RNN organizes memory by multiple update frequencies; CA3 implements explicit sequential progression of an event trace through a fixed chain.

**Reference:** Koutník et al. — *A Clockwork RNN*  
https://proceedings.mlr.press/v32/koutnik14

## 6. Event-triggered RL and Control

Event-triggered control asks whether control or communication needs to occur at every timestep:

\[
\text{continuous dynamics}\rightarrow\boxed{\text{intervene only when necessary}}.
\]

This supports the broader intuition that computation and policy intervention can be concentrated at informative or consequential events such as branch points or unexpected transitions.

**Difference:** event-triggered control usually concerns *when actions/control commands are updated*. IntrMotiv concerns *how internal state is represented between such events*.

**Reference:** Baumann et al. — *Deep Reinforcement Learning for Event-Triggered Control*  
https://arxiv.org/abs/1809.05152

## 7. Options and Temporal Abstraction

The options framework moves decision-making above the primitive-action timestep:

\[
\boxed{\text{decision}}
\rightarrow
\underbrace{\text{extended behavior}}_{\text{no new high-level decision}}
\rightarrow
\boxed{\text{decision}}.
\]

This is close to the intuition

\[
\boxed{\text{branch point}}
\rightarrow
\text{predictable trajectory segment}
\rightarrow
\boxed{\text{branch point}}.
\]

**Difference:** options create temporal abstraction mainly on the action side. DG→CA3 suggests a matching abstraction on the state/memory side:

\[
\text{sensory event}
\rightarrow
\underbrace{\text{cheap deterministic memory evolution}}_{\tau\text{ steps}}
\rightarrow
\text{sensory event}.
\]

**Reference:** Sutton, Precup, Singh — *Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning*  
https://www.sciencedirect.com/science/article/pii/S0004370299000521

# Synthesis: what is already established?

The IntrMotiv idea contains at least three ingredients with substantial prior art:

\[
\boxed{
\begin{array}{ll}
\textbf{A. Sparse/event-driven updates}
&\leftarrow\text{Phased LSTM, Skip RNN, event-triggered control}\\
\textbf{B. Fixed temporal dynamics}
&\leftarrow\text{reservoir computing}\\
\textbf{C. History as state}
&\leftarrow\text{PSRs and recurrent state representations}
\end{array}}
\]

None of these ingredients alone should be treated as the main novelty.

# What may still distinguish DG→CA3

The more specific combination is:

\[
\boxed{
o_t
\xrightarrow{\text{sparse DG}}
u_t
\xrightarrow[\text{fixed}]{x_{t+1}=Sx_t+Ju_t}
\text{high-dimensional explicit temporal state}
\xrightarrow{\text{RL}}
a_t.
}
\]

The particularly distinctive part is

\[
u_t=0
\quad\Rightarrow\quad
x_{t+\tau}=S^\tau x_t.
\]

When no informative input occurs, the representation:

- is not recomputed from scratch;
- is not updated by a learned recurrent network;
- is not simply frozen;
- instead **advances along a predetermined temporal trajectory**.

Elapsed time and event history are therefore represented largely **by construction rather than inference**.

This gives the sharper claim:

> **Replace repeated inference of temporal state with deterministic temporal expansion of sparse sensory events.**

Or:

\[
\boxed{
\text{learn WHAT happened}
+
\text{hard-code HOW memory evolves with time}
}
\]

instead of:

\[
\boxed{
\text{learn WHAT happened}
+
\text{learn WHAT to remember}
+
\text{learn HOW memory evolves}.
}
\]

# Implication for IntrMotiv

This framing may explain why the fixed CA3 mechanism works especially well under sparse DG input.

The advantage may not be that CA3 learns a richer instantaneous representation than an LSTM. Instead,

\[
\boxed{
\text{sparse sensory events}
+
\text{strong temporal prior}
\rightarrow
\text{useful dynamical state}
}
\]

while a generic recurrent model must infer the temporal transformation from data.

The strongest next literature comparison should therefore focus on:

- delay-line reservoirs;
- tapped-delay neural networks;
- Legendre Memory Units;
- HiPPO;
- state-space models;
- event-driven recurrent computation.

Those areas are likely to contain the strongest direct prior art against the claim that explicit structured temporal expansion can replace repeated learned inference.
