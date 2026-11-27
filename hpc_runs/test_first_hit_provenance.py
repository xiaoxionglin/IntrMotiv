"""Evidence integrity tests for source snapshots under concurrent edits."""

import json
from pathlib import Path
import tempfile
import unittest

from hpc_runs.first_hit_control.provenance import audit_source_snapshots, snapshot_sources


class FirstHitProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def source(self, relative, payload):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def test_duplicate_basenames_keep_distinct_exact_bytes(self):
        left = self.source("left/model.py", b"left\r\n\x00")
        right = self.source("right/model.py", b"right\n")
        destination = self.root / "archive"
        manifest = snapshot_sources([left, right], destination)
        self.assertNotEqual(manifest["sources"][0]["snapshot_path"], manifest["sources"][1]["snapshot_path"])
        for original, entry in zip((left, right), manifest["sources"]):
            self.assertEqual((destination / entry["snapshot_path"]).read_bytes(), original.read_bytes())
        result = audit_source_snapshots(destination)
        self.assertTrue(result["archive_valid"])
        self.assertTrue(result["current_sources_unchanged"])

    def test_existing_snapshot_is_never_overwritten(self):
        source = self.source("code.py", b"first")
        destination = self.root / "archive"
        manifest = snapshot_sources([source], destination)
        source.write_bytes(b"second")
        with self.assertRaises(FileExistsError):
            snapshot_sources([source], destination)
        self.assertEqual((destination / manifest["sources"][0]["snapshot_path"]).read_bytes(), b"first")

    def test_archive_tamper_is_not_confused_with_current_source(self):
        source = self.source("code.py", b"original")
        destination = self.root / "archive"
        manifest = snapshot_sources([source], destination)
        (destination / manifest["sources"][0]["snapshot_path"]).write_bytes(b"tampered")
        result = audit_source_snapshots(destination)
        self.assertFalse(result["archive_valid"])
        self.assertTrue(result["current_sources_unchanged"])

    def test_live_drift_preserves_archive_validity(self):
        source = self.source("code.py", b"original")
        destination = self.root / "archive"
        snapshot_sources([source], destination)
        source.write_bytes(b"concurrent edit")
        result = audit_source_snapshots(destination)
        self.assertTrue(result["archive_valid"])
        self.assertFalse(result["current_sources_unchanged"])
        self.assertTrue(result["sources"][0]["current_exists"])

    def test_missing_original_does_not_invalidate_saved_bytes(self):
        source = self.source("code.py", b"original")
        destination = self.root / "archive"
        snapshot_sources([source], destination)
        source.unlink()
        result = audit_source_snapshots(destination)
        self.assertTrue(result["archive_valid"])
        self.assertFalse(result["current_sources_unchanged"])
        self.assertFalse(result["sources"][0]["current_exists"])
        self.assertIsNone(audit_source_snapshots(destination, check_current=False)["current_sources_unchanged"])

    def test_manifest_traversal_is_rejected(self):
        source = self.source("code.py", b"original")
        destination = self.root / "archive"
        manifest = snapshot_sources([source], destination)
        manifest["sources"][0]["snapshot_path"] = "../code.py"
        (destination / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "contained"):
            audit_source_snapshots(destination)


if __name__ == "__main__":
    unittest.main()
