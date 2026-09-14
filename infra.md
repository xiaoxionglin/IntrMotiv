# Infrastructure Improvements

Track concrete infrastructure findings here so future threads can reuse
evidence and coordinate improvements. Consult this file before related work;
keep implementation guidance in the canonical workflow documents linked below.

## Open Improvements

### Portable runtime bootstrap — in progress

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
- **Status:** Python installed on G500; public dependencies installing. Private
  runtime transfer awaits user approval after an automatic review rejection.

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

None recorded yet.
