# Graph-Stabilized Orthogonal Recruitment

Date: 2026-09-03

## Status

The authoritative NEMO2 IntrMotiv checkout now implements graph-stabilized DG
recruitment behind `--dg_orthogonal_recruitment_mode=graph`. The default is
`legacy`, so historical commands retain their previous behavior. The 36-run
production manifest has been generated and audited but has not been submitted.

Five seed-99, 2M-frame ordinary Slurm preflights were submitted as jobs
`7976505` through `7976509`. They span direct, recovery, topology, both
half-lives, both redundancy thresholds, and the flat passive fallback. Their
outputs, caches, W&B data, and Slurm logs resolve below
`/work/classic/fr_xl1014-train`.

## Implemented rule

The original `L`-decision silent endpoint remains the proposal opportunity.
Graph mode changes only victim eligibility and choice:

1. A new field has birth support one and is protected while support is above
   `0.25`.
2. An edge exists when confidence is above `0.25` and elapsed time is positive.
3. A mature isolated vertex is eligible.
4. A mature vertex is also eligible if it is the lower-incident-confidence
   member of a bidirectionally supported pair with both elapsed times at most
   `D`; exact ties select the higher DG index.
5. Isolated vertices are preferred, then the lowest-support eligible vertex.
   If none is eligible, recruitment is skipped.

Policy-buffer HRL uses controllability confidence and `T_ctrl`. Flat and other
non-policy-buffer agents use a checkpointed passive graph. Passive edges come
from consecutive different exclusive DG events within `L` behavior decisions,
cross rollout boundaries, reset at physical episode boundaries, and never read
CA3. Representation-generation IDs reject evidence collected before a row was
reassigned.

Reassignment clears the affected row's optimizer and BatchNorm state, clears
incident passive and controllability edges, increments both representation
generations, restores birth support to one, and remains capped at one assignment
per accepted rollout. The old `recruitment_committed` buffer is retained as
ever-recruited telemetry but is no longer an eligibility gate in graph mode.

## Verification

- Focused implementation suite: 60 passed before the final evidence-preference
  test was added.
- Complete IntrMotiv suite: 146 passed, with 9 existing warnings.
- Graph-recruitment unit file after the additional policy-versus-passive test:
  12 passed.
- Real argument parsing succeeded for direct C05, topological C15, and flat
  passive-fallback configurations with the expected enlarged recurrent state.
- The audited production manifest contains 36 runs: corrected C05/C13/C15,
  `D` in `{4, 8}`, half-life in `{5k, 10k}`, and seeds `{8, 99, 123}`.

Production should remain held until all five preflights demonstrate learner
updates without runtime exceptions. Completed original C05/C13/C15 runs remain
contextual controls rather than contemporaneous matched runs.
