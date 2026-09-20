# Corridor study ASCII maps

Exact entity-layer bytes extracted from ../maps.json, with SHA-256 verified against the archive. These are the archived layouts used by corridor_geometry.study.json; no maps were regenerated.

Each file is 21 × 21 characters. `*` denotes a wall, a space denotes floor, and `P` denotes a permitted spawn cell (also floor). Openness is the wall-removal threshold: 0.00 corridor, 0.35 intermediate, 0.75 open-dominant. The same geometry is shared by all three architectures for each map seed and openness.

Keep spaces and the final newline intact; SHA256SUMS records the exact file hashes.
