# IntrMotiv recurrent DDQN/HER diagnostic

This package is a **frozen-reference local-control diagnostic**, not a completed
adaptive representation/manager architecture. The primary runtime supports the
selected F16 `dg_goal_input=none` parent. Write-conditioned memory and separate
target reconstruction have deterministic tests; write-conditioned collection
is explicitly rejected until actor-state rebuilding is integrated and tested.

The two arms share source decoder/FiLM weights, fresh Q initialization, permitted
observations, epsilon schedule, first-arrival reward, and nominal replay
intensity. They differ only in requested HER fraction (0 or 0.8). There is no
matched PPO learner in this package: parent zero-shot evaluation is a separate
reference, not a controlled PPO-versus-DDQN algorithm comparison.

- `checkpoint.py`: parent action/layout validation and conversion inventory.
- `features.py`: exact source normalization, frozen canonical DG, depth and map
  bypass; first-batch source-head parity check; no pose feature route.
- `contracts.py`: recognition, shift/injection, finite-budget reward, Double DQN.
- `replay.py`: bounded stream/episode replay with 71-step prefix validity,
  delayed future-HER eligibility, original-goal fallback and replay RNG state.
- `worker.py`, `batch.py`: current-parameter memory rebuild for both networks,
  masked Huber targets, hard copies including write modulation.
- `terminal.py`: explicit final-observation validity and deep-copy before reset.
- `train.py`: source environment, parallel collection, child counters,
  checkpoints and standard scalar layout. The physical episode decision clock
  is normalized by 1,800 without clipping; option budget by 64. These are input
  scales, not assertions about an engine's exact final decision number.
- `evaluate.py`: readout callback into the established exact-start matched
  evaluator; evaluation patch is supplied separately with regression tests.
  Grounded spatial destination qualification remains required.

Replay currently stores exact frozen preactivations and bypass outputs, not
frozen trunk features. Therefore it deliberately cannot support DG adaptation.
The working/reference encoder is entirely frozen in this stage. An adaptive
stage must change this feature contract, retain fixed reference labels, and
rebuild working memories from trunk features. Reusing the cached preactivations
would silently make the representation optimizer ineffective.

Invalid terminal observations are stored as `None`, excluded from TD/HER, and
counted. The same-step collector still receives the real reset observation for
the next physical episode. No previous image is substituted as a successor.
The supplied DMLab patch explicitly marks its cached terminal image invalid.

Checkpoint files omit replay content and are labeled
`warm_restart_requires_refill`; they are not exact-resume artifacts. The current
CLI refuses reusing an existing run namespace and does not implement restart.
Replay's own state-dict contract preserves storage and sampling RNG for a future
checksummed sidecar integration.

Run the independent tests with the project interpreter:

```bash
/home/xiaoxiong/miniforge3/envs/SF_git/bin/python -m unittest hpc_runs.test_intrmotiv_offpolicy
```

The runtime must be isolated from existing jobs. Use the reviewed terminal
contract patch and planner optimization in the selected new source checkout.
The deployment and submitted preflights are recorded in
`06_experiments/intrmotiv_ddqn_her_implementation_20260911.md`.
