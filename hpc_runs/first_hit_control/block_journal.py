"""Durable complete-block telemetry, with no partial analysis or resume path."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CompleteBlockJournal:
    """Append only complete declared Cartesian blocks; finalize once, never resume.

    Rows are validated and serialized before any bytes are appended. Every
    append fsyncs the data and atomically replaces a small progress receipt.
    Partial files are diagnostic only, even if some complete blocks exist.
    """
    def __init__(self, directory, blocks, row_keys, *, block_fields=("source", "repeat"),
                 row_fields=("command", "target"), final_name="candidate_trials.jsonl"):
        self.directory = Path(directory)
        self.blocks = [tuple(b) for b in blocks]
        self.row_keys = {tuple(k) for k in row_keys}
        if not self.blocks or len(set(self.blocks)) != len(self.blocks) or not self.row_keys:
            raise ValueError("Require unique nonempty declared blocks and row keys")
        self.block_fields, self.row_fields = tuple(block_fields), tuple(row_fields)
        if Path(final_name).name != final_name:
            raise ValueError("Journal final name must be one local filename")
        self.final = self.directory / final_name
        self.partial = self.directory / (final_name + ".partial")
        self.status = self.directory / "block_journal.json"
        if any(path.exists() for path in (self.final, self.partial, self.status)):
            raise FileExistsError("No overwrite or automatic resume of block journals")
        self.completed, self.rows, self.digest = [], 0, hashlib.sha256()
        self.partial.touch(exist_ok=False)
        self._receipt("partial")

    def _receipt(self, status):
        receipt = dict(schema="intrmotiv/complete-block-journal/v1", status=status,
                       updated_utc=datetime.now(timezone.utc).isoformat(),
                       aggregatable=status == "complete", automatic_resume=False,
                       expected_blocks=[list(b) for b in self.blocks],
                       completed_blocks=[list(b) for b in self.completed], rows=self.rows,
                       expected_rows_per_block=len(self.row_keys), sha256=self.digest.hexdigest(),
                       filename=self.final.name if status == "complete" else self.partial.name)
        temporary = self.status.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.status)
        descriptor = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return receipt

    def append(self, block, rows):
        block = tuple(block)
        if len(self.completed) >= len(self.blocks) or block != self.blocks[len(self.completed)]:
            raise ValueError("Journal block is duplicate, missing, or out of declared order")
        keys = [tuple(row[field] for field in self.row_fields) for row in rows]
        if (len(keys) != len(self.row_keys) or set(keys) != self.row_keys
                or any(tuple(row[field] for field in self.block_fields) != block for row in rows)):
            raise ValueError("Only a complete declared block may be persisted")
        payload = "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows).encode()
        with self.partial.open("ab") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        self.digest.update(payload)
        self.completed.append(block)
        self.rows += len(rows)
        return self._receipt("partial")

    def finalize(self):
        if self.completed != self.blocks:
            raise ValueError("Cannot finalize a journal with missing declared blocks")
        if _hash_file(self.partial) != self.digest.hexdigest():
            raise ValueError("Journal bytes changed before finalization")
        os.replace(self.partial, self.final)
        return self._receipt("complete")


def validate_complete_journal(directory, expected_blocks, expected_rows_per_block, expected_hash):
    """Pure receipt/byte validation; incomplete journals never enter analysis."""
    directory = Path(directory)
    receipt = json.loads((directory / "block_journal.json").read_text())
    expected = [list(block) for block in expected_blocks]
    if (receipt.get("schema") != "intrmotiv/complete-block-journal/v1"
            or receipt.get("status") != "complete" or receipt.get("aggregatable") is not True
            or receipt.get("automatic_resume") is not False):
        raise ValueError("Incomplete journal is diagnostic only, never aggregatable")
    if (receipt["expected_blocks"] != expected or receipt["completed_blocks"] != expected
            or receipt["expected_rows_per_block"] != expected_rows_per_block
            or receipt["rows"] != len(expected) * expected_rows_per_block):
        raise ValueError("Completed journal omits or changes declared blocks")
    filename = receipt["filename"]
    if Path(filename).name != filename or filename != "candidate_trials.jsonl":
        raise ValueError("Unexpected final candidate journal file")
    if receipt["sha256"] != expected_hash or _hash_file(directory / filename) != expected_hash:
        raise ValueError("Completed journal hash differs from final trial receipt")
    return receipt
