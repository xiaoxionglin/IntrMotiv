# Infrastructure Improvements

Track concrete infrastructure findings here so future threads can reuse
evidence and coordinate improvements. Consult this file before related work;
keep implementation guidance in the canonical workflow documents linked below.

## Open Improvements

### Explicit depth preprocessing contract — proposed

- **Evidence:** The local runtime `DepthEncoder` consumes the RGBD channel,
  and both the standard encoder and `hpc_runs/intrmotiv_offpolicy/features.py`
  reuse it. Its capped inverse response now assumes unnormalized depth codes;
  study configurations use `normalize_input=False`, while DMLab defaults expose
  image normalization settings.
- **Impact:** Enabling image normalization could silently change the meaning
  of the depth response. Depth codes also should not be described as calibrated
  world distances without checking the renderer's conversion.
- **Proposed improvement:** Make the shared RGB/depth preprocessing contract
  explicit and validate it at encoder construction before supporting other
  normalization configurations.
- **Acceptance criteria:** Tests cover supported normalization settings and
  reject or correctly convert incompatible depth inputs.
- **Status:** Proposed; the requested inverse response is implemented locally
  in `/home/xiaoxiong/SFgit/SF_hipposlam`, with two focused tests passing.
- **Reusable lesson:** Locate runtime source through the canonical setup record
  and search source-file types first. An unrestricted vault search pulled in
  large saved experiment artifacts. Reusing `DepthEncoder` covers both consumers
  without duplicate transforms. Authoritative verification:
  `python -m pytest -q -p no:cacheprovider sf_working_directories/IntrMotiv/tests/test_depth_encoder.py`
  from the runtime checkout using the `SF_git` Python environment.

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

None recorded yet.
