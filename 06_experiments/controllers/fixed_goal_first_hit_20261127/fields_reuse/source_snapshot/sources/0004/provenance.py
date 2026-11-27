"""Archive exact experiment source bytes before execution.

Archive integrity and subsequent changes to working files are separate facts.
An experiment remains reproducible from an intact archive even if a concurrent
agent changes its original source paths after the snapshot was taken.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


SCHEMA = "intrmotiv/source-snapshot/v1"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snapshot_sources(paths, destination):
    """Copy exact bytes into a new directory and return the saved manifest.

    Each input gets its own numbered subdirectory, so files with equal basenames
    cannot collide. Existing destinations are refused rather than reused or
    overwritten. The caller must invoke this before starting the experiment and
    record the returned manifest in its run provenance.
    """
    originals = [Path(path).resolve() for path in paths]
    if not originals or len(set(originals)) != len(originals):
        raise ValueError("Source paths must be nonempty and unique")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Source snapshot destination already exists: {destination}")
    # Read once; the hashed content is exactly what is archived even if an
    # original changes between this read and the later experiment.
    payloads = [(path, path.read_bytes()) for path in originals]
    destination.mkdir(parents=True, exist_ok=False)
    entries = []
    for index, (original, data) in enumerate(payloads):
        relative = Path("sources") / f"{index:04d}" / original.name
        archived = destination / relative
        archived.parent.mkdir(parents=True, exist_ok=False)
        archived.write_bytes(data)
        entries.append({"source_path": str(original), "snapshot_path": relative.as_posix(),
                        "sha256": _sha256(data), "bytes": len(data)})
    manifest = {"schema": SCHEMA, "sources": entries}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def audit_source_snapshots(destination, *, check_current=True):
    """Check saved source integrity and report original-file drift separately.

    Missing or altered archives make ``archive_valid`` false. Missing or altered
    originals only affect ``current_sources_unchanged``. Malformed manifests or
    paths escaping the archive raise ValueError rather than reading elsewhere.
    """
    destination = Path(destination).resolve()
    manifest = json.loads((destination / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or not isinstance(manifest.get("sources"), list) or not manifest["sources"]:
        raise ValueError("Invalid source snapshot manifest")
    results = []
    seen = set()
    for entry in manifest["sources"]:
        relative = Path(entry["snapshot_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in seen:
            raise ValueError("Snapshot paths must be unique and contained in the archive")
        seen.add(relative.as_posix())
        archived = destination / relative
        # Resolve existing links too: a corrupted archive must not redirect the
        # integrity reader outside its root.
        if not archived.resolve().is_relative_to(destination):
            raise ValueError("Snapshot path escapes archive")
        snapshot_exists = archived.is_file()
        snapshot_hash = _sha256(archived.read_bytes()) if snapshot_exists else None
        current = Path(entry["source_path"])
        result = {"source_path": str(current), "snapshot_path": relative.as_posix(),
                  "expected_sha256": entry["sha256"], "snapshot_sha256": snapshot_hash,
                  "snapshot_valid": snapshot_exists and snapshot_hash == entry["sha256"]}
        if check_current:
            current_exists = current.is_file()
            current_hash = _sha256(current.read_bytes()) if current_exists else None
            result.update(current_exists=current_exists, current_sha256=current_hash,
                          current_matches_snapshot=current_hash == entry["sha256"])
        results.append(result)
    return {"schema": SCHEMA, "archive_valid": all(row["snapshot_valid"] for row in results),
            "current_sources_unchanged": all(row["current_matches_snapshot"] for row in results) if check_current else None,
            "sources": results}
