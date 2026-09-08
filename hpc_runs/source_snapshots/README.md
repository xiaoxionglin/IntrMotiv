# Scoped runtime source snapshots

`persistent_intrinsic_control_hotfix_20260908.tar.gz` preserves the complete
Persistent Intrinsic Control runtime source after the legacy-recruitment /
policy-graph invalidation hotfix. It retains the original implementation files
and adds the focused regression, forced-replacement preflight, and exact
W_REF_JOINT seed-123 retry adapter. The pre-hotfix
`persistent_intrinsic_control_20260908.tar.gz` remains unchanged.

Hotfix archive SHA-256:
`c01a2550046233cd9b1474a6d043bdaf9f29906947f60e79b418dae681746afd`.

`ca3_memory_novelty_goal_20260907.tar.gz` retains the exact seventeen modified
or added IntrMotiv runtime files for the CA3 finite-memory batch. It is a
102-KiB reproducibility artifact, not a second maintained source checkout.
The deployment target is the existing NEMO2 SF_hipposlam checkout; Sample Factory
and DMLab dependencies are not bundled. Studies and thin audit/smoke adapters
are versioned normally in the surrounding `hpc_runs/` directory.

Archive SHA-256:
`83d5e265d92e78bba211bccca18c996a98deaab0183269f4e4c79423f18c3ea9`.

Inspect or extract into a separate temporary directory when reviewing. Do not
blindly unpack over an evolving runtime checkout: full modified files also
contain pre-existing project changes that must be compared and preserved.
See `06_experiments/ca3_memory_novelty_goal_implementation.md` for semantics,
tests, source provenance, and actual submission records.
