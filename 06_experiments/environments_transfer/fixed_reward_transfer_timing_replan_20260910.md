# Matched-timing transfer replan

Status: replacement authorized. The old transfer batch is cancelled; progress
and job IDs are recorded in the
[repeat-8 launch record](fixed_reward_transfer_repeat8_launch_20260910.md).

## Immediate recommendation

Rerun downstream transfer at frame repeat 8 with the historical reduced
five-action interface. Both selected SCR and SAT checkpoints were trained with
that interface. This corrects the timing mismatch without requiring new
pretraining or introducing an action-space change.

Retain the seven existing conditions: scratch, and frozen DG, tuned DG, and
policy transfer from each source. Retain downstream seeds 42, 1234, and 9999,
giving 21 runs with a 30-hour training budget and 32-hour Slurm allocation.
Use the same source checkpoint files and hashes as the original study.

The repeat-8 scratch control is necessary. The existing repeat-4 scratch runs
remain a secondary timing reference; they cannot replace the matched control.
The current algorithm screen also uses eight actions, so the old five-action
scratch control is not an exact control for that screen either.

## What this rerun establishes

For this timing comparison, retain the current transfer scopes, task vector,
normalization rules, optimizer settings, and reward routing. Change only frame
repeat and run/output/logging identifiers. This isolates timing within each
condition. It does not resolve the separate question of whether replacing a
waypoint one-hot with a learned constant preserves useful goal control.

Compare transfer against scratch within each repeat setting at common frame
budgets; also report policy decisions and wall-clock time. Across repeats,
equal simulator frames do not imply equal numbers of policy decisions. Treat
existing normalization and goal-interface differences as explicit limitations
when interpreting frozen-versus-tuned or DG-versus-policy comparisons.

Before submission, derive a new canonical StudySpec from the original, verify
that its expanded commands differ only in declared settings, and preserve the
schema, workflow version, and new SHA-256. Validate actual source configs and
the instantiated environment's ordered action vectors, repeat, observations,
memory settings, conditioning module, and transferred tensor inventory. Run
the required preflight and launcher print-only/submission audits in a new
workspace namespace. Retain an immutable initialization artifact so checkpoint
rotation cannot remove the evidence needed for exact loading checks.

## Existing repeat-4 alternatives

The canonical navigation8 algorithm screen contains 18 running repeat-4 runs
(six families, seeds 8, 99, and 123). Actual saved configurations confirm
navigation actions enabled and reduced actions disabled. At inspection, the
fresh SCR family had checkpoints at approximately 56–78M frames, and the fresh
SAT family at 60–77M frames. Both use FiLM conditioning. These are candidates,
not yet qualified transfer winners.

An eventual repeat-4 transfer study should retain their eight ordered actions
and add a matching eight-action scratch control. Select sources using
pretraining evidence and a declared checkpoint budget, then verify spatial
representation and goal control with the established evaluator. The W_REF
families have repeat-8 representation ancestry and additive conditioning;
they should not be presented as fresh repeat-4 FiLM pretraining controls.

## Provenance and reusable lesson

Original transfer StudySpec: `hpc_runs/studies/fixed_reward_transfer.study.json`,
schema `intrmotiv/study/v1`, workflow 1.5.0, SHA-256
`b7e5c5db2547cf5e05763718479dfa8a54e9dbd83a6b30012d33a9c0c2920d73`.

Navigation8 StudySpec: `hpc_runs/studies/navigation8_algorithm_screen.study.json`,
schema `intrmotiv/study/v1`, workflow 1.5.0, SHA-256
`c435ddac609945336d1e42ca16ed0bcc8fd2d46be13eef167a0085b39b682094`.

The authoritative evidence is the expanded StudySpec together with actual
run configs and saved checkpoints. Checking frame repeat alone misses action
semantics. Check both before ranking source runs or authoring a transfer matrix;
reuse the canonical metric collector rather than rediscovering historical
launch scripts. A successful loading preflight is not a behavioral transfer
qualification.
