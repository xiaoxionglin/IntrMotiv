# Easy Landmark-Maze Architecture Screen — Implementation Record

## Status

Local implementation and native qualification tooling are complete. Production
is **not released or submitted**. NEMO2 synchronization, focused tests, the six
2M qualification runs, frozen evaluation, checkpoint reload certificates, and
the canonical submission audit remain hard gates.

Workflow: `1.12.0`; study schema: `intrmotiv/study/v1`; map geometry schemas:
backward-compatible `intrmotiv/map-geometry/v1` and cue-aware
`intrmotiv/map-geometry/v2`.

## Reviewed identities

| Artifact | Count | SHA-256 |
|---|---:|---|
| Rich production StudySpec | 9 runs | `0c27a8a65a8f7a927e8607f3ad844ab09e7ee747b3a49482f23b01d5bf83af00` |
| No-cue control StudySpec | 3 runs | `cc94375ea0bd3c046470c5b868c1d9e497677b08b8971d38026a100988837862` |
| Qualification StudySpec | 6 runs | `32e35541ca2808a3264b57da08dd78f7abbbb88c9f68ecb1c2aef38548029411` |
| Geometry archive file | 2 modes | `54c2c464f48d55e9c1485c31e74076a3c643dd01d06334108dcc593d623df671` |
| Entity layer | rich and none | `1c70fc2595912dc1077601e9b1cae808555216ad3f74e9b86090736e89da3db9` |
| Rich cue layout | 20 rendered sites | `69bc976ebd4c108d541184716233484d259ead4a3fa455490ceefe45db171191` |
| Control cue layout | same 20 unrendered sites | `14111dca4cded7489bac13e5bc2d68cbc54a5aba135e2162cc65e0c60b49dc7a` |

The runtime is isolated at
`/home/xiaoxiong/SFgit/SF_hipposlam_easy_landmark_20260923` on
`codex/easy-landmark-maze-20260923`, based on commit
`c002faff2c6832f9b0ce63bc6401f95e9cb4718d`. Exact source-tree and dirty-state
hashes are in [runtime_source_provenance.json](easy_landmark_maze_20260923/runtime_source_provenance.json).
The qualified corridor checkout was copied, not edited.

## Implemented contracts

- `easy_landmark_maze_noreward` fixes geometry seed 1001, wall-removal
  probability 0.85, layout seed 20260923, 120-second episodes, zero external reward,
  frameskip 4, and navigation8 actions.
- The resulting 21-by-21 entity layer has 337 accessible cells: an open field
  with sparse retained obstacles rather than a perfect maze.
- A portable LCG/Fisher-Yates selector shared by Python and Lua samples 20
  distinct wall faces without replacement and also requires 20 distinct
  adjacent accessible cells. D01–D10 use fixed existing decals; C01–C10 use a
  fixed saturated palette. Other walls use one neutral gray texture.
- Rich and none modes share entity geometry, spawn behavior, timing, and site
  order. None keeps every reserved site in privileged telemetry but renders it
  neutral.
- Native entity and cue manifests are verified against the archive on reset.
  Entity, cue manifest, and debug pose are removed before policy observation
  normalization. The policy receives no position or cue identity.
- Map geometry v2 adds optional cue arrays while the v1 reader remains valid.
  Online/offline NPZ schema v1 accepts the optional additions.
- Spatial analysis adds cue visitation, traversable-geodesic nearest-cue
  distance, minimum-cost one-to-one cue/peak assignments with unmatched rows,
  raw and capacity-normalized coverage, decal/color subsets, CSV export, and
  cue overlays on occupancy, trajectory, and field figures.
- Standard 10k field rows are kept separate from intervention-only inventory:
  21 rich rows and 15 control rows. Production intervention inventory contains
  18 rich rows; control contains 6.
- SCR ARR DIRS and DGP HIT JOINT LEG preserve their parent arguments. Waypoint
  uses F64, DDQN+HER, stored replay, cadence 2048, and the existing 1024 TD plus
  1024 HER split. Every run stores its exact parent fingerprint and a
  machine-readable override set.

## Local verification

- 51 canonical geometry/workflow tests passed before runtime staging.
- Runtime tests pass policy-privacy stripping, exact rich/control reserved-site
  identity, native manifest verification, repeated-reset equality, zero reward,
  and the 1800-decision timeout at frameskip 4.
- All six qualification commands parse through `parse_dmlab_args`; the Waypoint
  rows resolve to stored DDQN+HER with cadence 2048 and the 1024/1024 split.
- Native rendering compiled all 20 review approaches. Visual inspection found
  10 unique, centered, unclipped decals; 10 visibly distinct colored wall
  faces; neutral-gray non-cue walls; consistent lighting; and no cue-face
  clipping. The review renderer is isolated from the production level.

![Twenty native cue approaches](easy_landmark_maze_20260923/figures/cue_approaches.png)

![Entity map and reserved cue faces](easy_landmark_maze_20260923/figures/map_preview.png)

## Release gates

1. Recompute runtime provenance hashes, synchronize workflow, StudySpecs,
   runtime source, DMLab patch, and isolated runfiles to the active
   `/work/classic/fr_xl1014-corridor-geometry` allocation.
2. Run the focused workflow and runtime tests on NEMO2, then render and inspect
   all three plans. Keep all cache, W&B, checkpoint, log, temporary, and
   analysis paths under the active allocation.
3. Run the six 2M qualification rows. Require finite learning, correct
   DG/controller ownership, snapshots at 1M and 2M, exact reload certificates,
   frozen-state evaluation, and finite cue-aware metrics.
4. Run the canonical qualification audit. Archive the StudySpec hashes,
   rendered plan, runtime/source hashes, jobs manifest, and audit output.
5. Only after all rows pass, use the print-only production launcher and
   canonical submission audit. Do not submit either production StudySpec on a
   partial qualification.

## Analysis guardrails

Report localization, exploration, and connectivity/control independently;
retain per-seed results and avoid significance claims from three seeds. The
seed-99 rich-minus-none comparison is a paired single-seed causal check, not
replicated evidence. Twenty cues exceed SCR/DGP F16 capacity, so the relevant
question is how their code partitions the maze, not whether every cue receives
one DG unit. No composite score is defined.

## Reusable experience

The authoritative checks were the native Lua manifest, the actual DMLab reset,
and the authoritative training parser. They exposed two issues invisible to
pure Python tests: corridor-only settings passed to the new level, and a shared
wrapper that assumed every archived map had cue metadata. The efficient future
path is to extend the canonical geometry artifact first, test both schema
versions, parse every rendered qualification row, then compile one rich and one
neutral native map before rendering the full visual review set.
