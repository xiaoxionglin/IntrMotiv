# Contextual Landmark State Design

**Status:** Deferred design direction; preserve for planning after the minimal
recruitment/control update.

## Problem

Some DG units have spatially scattered rate fields and become common sink
vertices in the learned transition graph. Such a vertex is not merely a poor
place field: its different activations can denote behaviorally different
states, so transitions into the shared DG identity create false graph
convergence and an ill-defined control target.

The transferable requirement is therefore not Euclidean unimodality itself:

> A control landmark must denote a single transitionally coherent contextual
> state.

## Direction A: contextual CA3 graph states

Keep DG units as visual event detectors, but define a graph state from the
current DG event, the first `R` positions of the preceding CA3 state, and the
previous action. The short `R` window captures arrival context without making
the full `L`-step trace part of identity.

This is the simplest diagnostic because it adds no learned recurrent path. Its
main risks are state proliferation and splitting one physical place by arrival
route. It also changes the interpretation: DG units are observation features,
whereas contextual CA3 patterns are the actual landmarks.

## Direction B: CA3-conditioned DG landmarks

Retain individual DG units as the landmark identities, but condition their
activation on one-step-delayed CA3/action context:

```text
sensory proposal = DG_projection(visual_features)
context signal   = context_adapter(stop_gradient(CA3[:R]), previous_action)
DG activation    = threshold(sensory proposal + context signal)
```

Use behavior-time CA3 context during replay and do not backpropagate through
the stored CA3 recurrence. Control success, graph degree, and recruitment
eligibility must not enter this feedback path.

Adding feedback alone does not resolve aliasing. The corresponding learning
principle should be action-conditioned transition predictability: occurrences
assigned to one DG unit should have compatible successor distributions under
the same action. This is meaningful in physical, web, and abstract-graph
domains.

## Diagnostic before implementation

1. Collect occurrence-level telemetry for problematic sink units, including
   position, the first `R` CA3 positions, previous action, and successor event.
2. Plot each sink's rate map conditioned on recent CA3/action context.
3. Compare action-conditioned successor distributions between contexts.
4. If context separates the spatial components cleanly, test contextual graph
   states first. If context is predictive but fragments excessively, test the
   detached CA3-to-DG context adapter. If neither works, longer memory or path
   integration is genuinely required.

## Experimental boundary

Learn and inspect the contextual representation before training control, then
freeze landmark identity for the control phase. Do not combine this experiment
with graph-dependent replacement recruitment.

## Near-term directional recruitment hypothesis

The smaller interim proposal is to distinguish three structural cases using a
stable passive event graph:

- no incoming and no outgoing support: unused/isolated candidate;
- incoming support but no outgoing support: sink/alias candidate;
- short, supported edges in both directions: possible duplicate pair.

Outgoing support is the relevant protection against sink behavior. High or
short incoming support alone should not be called redundancy: it can also
describe a legitimate hub. Therefore sink eligibility should require weak
outgoing support, while duplicate eligibility should retain the bidirectional
short-edge test.
