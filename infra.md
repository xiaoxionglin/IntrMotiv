# Infrastructure Improvements

Track concrete infrastructure findings here so future threads can reuse
evidence and coordinate improvements. Consult this file before related work;
keep implementation guidance in the canonical workflow documents linked below.

## Open Improvements

### Portable runtime bootstrap — verified; patcher cleanup proposed

- **Evidence:** G500 setup found a nonportable absolute DMLab wheel path in the
  runtime `requirements.txt`, plus a conda-only guard in the DMLab asset patcher.
  The desktop environment also reports torch/torchvision versions that are not
  an official matched pair. See the [setup record](04_implementation/g500_environment.md).
- **Impact:** Blind environment copying or requirements installation is not a
  reliable way to prepare a second GPU host.
- **Proposed improvement:** Reuse `setup.py`, the existing DMLab wheel and asset
  patcher, and a matched GPU package pair; record the verified environment and
  make the asset patcher accept an explicitly selected Python environment.
- **Acceptance criteria:** `pip check`, GPU/ResNet inference, custom DMLab
  reset/step, and focused runtime tests pass in an isolated target environment.
- **Status:** G500 environment complete after explicit source-transfer approval.
  All listed acceptance checks passed, including 68 focused tests. The existing
  patcher was reused with a command-scoped `CONDA_PREFIX`; removing its conda-only
  guard remains a proposed runtime-source cleanup.
- **Outcome:** A fresh matched CUDA environment and user-local Ubuntu SDL2 library
  worked. `ldd` isolated the only missing native dependency. Reuse the recorded
  activation helper, resolved package manifest and smoke script next time.

For each finding, record:

- **Title and status:** proposed, in progress, blocked, or completed.
- **Evidence:** observed failure, duplication, bottleneck, or readability issue,
  with relevant file links or commands.
- **Impact:** how it affects users, future threads, reliability, or efficiency.
- **Proposed improvement:** the smallest reusable change, including existing
  modules or packages considered.
- **Acceptance criteria:** observable checks that demonstrate improvement.
- **Outcome:** changes made, verification results, and reusable lessons.

## Canonical References

- [Agent guidelines](AGENTS.md)
- [Standardized study workflow](04_implementation/standardized_study_workflow.md)
- [Reusable place-field telemetry](04_implementation/reusable_place_field_telemetry.md)

## Completed Improvements

### Explicit depth preprocessing contract — completed locally, 2026-09-14

- **Evidence:** `sample_factory/utils/normalize.py` applies fixed observation
  scaling even with `normalize_input=False`. The initial hard-coded inverse
  transform missed this and also changed historical configuration behavior.
- **Impact:** Depth units and old-policy inputs could change silently.
- **Improvement:** Shared `DepthEncoder` now defaults to legacy pass-through,
  with a single inverse switch and fixed gain; inverse mode restores fixed scaling
  and rejects running normalization of depth. See the
  [architecture reference](04_implementation/current_hrl_architecture_summary.md#optional-inverse-depth-response-local-runtime-2026-09-14)
  for parameters and usage.
- **Acceptance criteria and outcome:** 17 focused tests passed, covering old
  configs, exact legacy sampling, unchanged checkpoint state, CLI toggles,
  fixed preprocessing, saved mode/gain compatibility, explicit-switch precedence,
  invalid saved gains, normalization rejection, and norms.
  Implemented only in `/home/xiaoxiong/SFgit/SF_hipposlam`.
- **Reusable lesson:** Trace the full preprocessing path before changing an
  observation transform; test with the real normalizer. Reference-distance
  scaling is a heuristic until actual feature norms are measured. Search source
  file types first to avoid large saved experiment artifacts, and reuse the
  shared encoder rather than duplicating transforms across consumers. Expose
  only the experimental choice needed for new runs; keep arbitrary scaling
  constants fixed and migration support confined to saved-config loading.
- **Authoritative check:** Run
  `python -m pytest -q -p no:cacheprovider sf_working_directories/IntrMotiv/tests/test_depth_encoder.py`
  from the runtime checkout using the `SF_git` Python environment.
