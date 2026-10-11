"""Parse the declared COSYNE condition identity used by analysis adapters."""

from __future__ import annotations


def factors(condition: str) -> tuple[str, str, str]:
    """Return family, encoder timing, and worker timing from a validated name."""
    pieces = condition.split("_")
    if len(pieces) < 3 or pieces[0] != "COS27" or pieces[1] not in {"C01", "C05", "C15"}:
        raise ValueError(f"Unexpected COSYNE condition {condition!r}")
    family = pieces[1]
    encoder = pieces[2].lower()
    if encoder not in {"temporal", "constant"}:
        raise ValueError(f"Unexpected encoder timing {condition!r}")
    if family == "C05":
        if len(pieces) != 4 or pieces[3].lower() not in {"temporal", "constant"}:
            raise ValueError(f"Unexpected C05 worker timing {condition!r}")
        worker = pieces[3].lower()
    else:
        if len(pieces) != 3:
            raise ValueError(f"Unexpected non-C05 condition {condition!r}")
        worker = "inapplicable" if family == "C01" else "temporal"
    return family, encoder, worker
