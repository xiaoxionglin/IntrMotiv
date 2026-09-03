"""Command-line entry point for the canonical IntrMotiv study workflow."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .analysis import linear_contrasts, summarize_records
from .spec import SpecError, StudySpec, load_study
from .telemetry import (
    build_place_field_manifests,
    discover_nemo_checkpoints,
    write_manifest,
)
from .tensorboard import collect_online_records


def _write_json(path: Path | None, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(rendered, end="")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        raise SpecError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in values:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _metric_names(study: StudySpec) -> list[str]:
    metrics: list[str] = []
    for field in ("window_metrics", "cumulative_metrics"):
        mapping = study.analysis.get(field, {})
        if not isinstance(mapping, Mapping):
            raise SpecError(f"analysis.{field} must be an object")
        metrics.extend(mapping)
    if not metrics:
        raise SpecError("analysis defines no metrics")
    return metrics


def _normalize_records(study: StudySpec, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restore typed design metadata and verify one row per declared run."""

    expected = {run.name: run for run in study.expand_runs()}
    observed_names = [record.get("run_name") for record in records]
    if any(not isinstance(name, str) for name in observed_names):
        raise SpecError("every analysis row must contain run_name")
    if len(observed_names) != len(set(observed_names)):
        raise SpecError("analysis input contains duplicate run_name rows")
    observed = set(observed_names)
    if observed != set(expected):
        raise SpecError(
            f"analysis run matrix differs from the study; "
            f"missing={sorted(set(expected) - observed)}, unexpected={sorted(observed - set(expected))}"
        )
    normalized: list[dict[str, Any]] = []
    for record in records:
        run = expected[record["run_name"]]
        normalized.append({
            **record,
            "condition": run.condition,
            "base": run.base,
            "seed": run.seed,
            **run.factors,
            **run.metadata,
        })
    return normalized


def _analyze(study: StudySpec, records: list[dict[str, Any]], output_dir: Path) -> None:
    records = _normalize_records(study, records)
    group_by = study.analysis.get("group_by", ["condition"])
    contrast_group_by = study.analysis.get("contrast_group_by", group_by)
    replicate_by = study.analysis.get("replicate_by", ["seed"])
    contrasts = study.analysis.get("contrasts", [])
    if not all(
        isinstance(item, list)
        for item in (group_by, contrast_group_by, replicate_by, contrasts)
    ):
        raise SpecError("analysis grouping fields and contrasts must be arrays")
    metrics = _metric_names(study)
    summary = summarize_records(records, group_by, metrics)
    _write_csv(output_dir / "condition_summary.csv", summary)
    if contrasts:
        detailed, contrast_summary = linear_contrasts(
            records, metrics, contrast_group_by, replicate_by, contrasts
        )
        _write_csv(output_dir / "paired_contrasts.csv", detailed)
        _write_csv(output_dir / "paired_contrast_summary.csv", contrast_summary)
    _write_json(output_dir / "analysis_manifest.json", {
        **study.provenance(),
        "input_rows": len(records),
        "metrics": metrics,
        "group_by": group_by,
        "contrast_group_by": contrast_group_by,
        "replicate_by": replicate_by,
        "contrasts": [contrast.get("name") for contrast in contrasts],
    })


def command_validate(args: argparse.Namespace) -> None:
    study = load_study(args.study)
    _write_json(None, {**study.provenance(), "status": "valid"})


def command_render_runs(args: argparse.Namespace) -> None:
    study = load_study(args.study)
    _write_json(args.output, {
        **study.provenance(),
        "rows": [run.as_dict() for run in study.expand_runs()],
    })


def command_collect_online(args: argparse.Namespace) -> None:
    study = load_study(args.study)
    fixed_window = None
    if args.window_low is not None or args.window_high is not None:
        if args.window_low is None or args.window_high is None:
            raise SpecError("--window-low and --window-high must be supplied together")
        fixed_window = (args.window_low, args.window_high)
    records = collect_online_records(study, args.batch_root, fixed_window=fixed_window)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "per_run.csv", records)
    _analyze(study, records, args.output_dir)


def command_analyze_csv(args: argparse.Namespace) -> None:
    study = load_study(args.study)
    records = _read_csv(args.input)
    if len(records) != study.expected_runs:
        raise SpecError(
            f"analysis input has {len(records)} rows; study expects {study.expected_runs}"
        )
    _analyze(study, records, args.output_dir)


def command_render_telemetry(args: argparse.Namespace) -> None:
    study = load_study(args.study)
    inventory = discover_nemo_checkpoints(study, args.batch_root)
    rows, trajectory = build_place_field_manifests(study, inventory)
    write_manifest(args.output_root / "analysis_manifest.tsv", rows)
    write_manifest(args.output_root / "trajectory_manifest.tsv", trajectory)
    _write_json(args.output_root / "study_manifest.json", {
        **study.provenance(),
        "telemetry_protocol": study.telemetry.get("protocol"),
        "analysis_rows": len(rows),
        "trajectory_rows": len(trajectory),
    })
    print(
        f"wrote {len(rows)} telemetry rows and {len(trajectory)} trajectory rows "
        f"to {args.output_root}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate and fingerprint a study")
    validate.add_argument("study", type=Path)
    validate.set_defaults(func=command_validate)

    render = subparsers.add_parser("render-runs", help="render the complete training matrix")
    render.add_argument("study", type=Path)
    render.add_argument("--output", type=Path)
    render.set_defaults(func=command_render_runs)

    collect = subparsers.add_parser(
        "collect-online", help="collect TensorBoard rows and run standard analysis"
    )
    collect.add_argument("study", type=Path)
    collect.add_argument("batch_root", type=Path)
    collect.add_argument("output_dir", type=Path)
    collect.add_argument("--window-low", type=int)
    collect.add_argument("--window-high", type=int)
    collect.set_defaults(func=command_collect_online)

    analyze = subparsers.add_parser("analyze-csv", help="analyze an existing per-run CSV")
    analyze.add_argument("study", type=Path)
    analyze.add_argument("input", type=Path)
    analyze.add_argument("output_dir", type=Path)
    analyze.set_defaults(func=command_analyze_csv)

    telemetry = subparsers.add_parser(
        "render-telemetry", help="discover checkpoints and write standard telemetry manifests"
    )
    telemetry.add_argument("study", type=Path)
    telemetry.add_argument("batch_root", type=Path)
    telemetry.add_argument("output_root", type=Path)
    telemetry.set_defaults(func=command_render_telemetry)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (SpecError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
