"""Build the current Overleaf project or its expanded figure library.

The default archive contains the current figure sources, PDFs, and the active
Figure 2 layout trial. ``--full`` writes a separate archive with candidate
COSYNE plots; it never replaces the current upload package.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path


DRAFT_DIR = Path(__file__).resolve().parent
REPO_ROOT = DRAFT_DIR.parents[1]
RESULTS_DIR = REPO_ROOT / "06_experiments" / "results"
ARCHIVE_PATH = DRAFT_DIR / "cosyne_2027_overleaf_sources.zip"
FULL_ARCHIVE_PATH = DRAFT_DIR / "cosyne_2027_overleaf_sources_full.zip"
LAYOUT_TRIAL_ASSETS = {
    "figures/Figure2_fiddle.svg": DRAFT_DIR / "Figure2_fiddle.svg",
    "figures/Figure2_tetris.svg": DRAFT_DIR / "Figure2_tetris.svg",
    "figures/Figure2_tetris.pdf": DRAFT_DIR / "Figure2_tetris.pdf",
    "figures/Figure2_tetris.png": DRAFT_DIR / "Figure2_tetris.png",
    "figures/Figure2_tetris_sidebar.svg": DRAFT_DIR / "Figure2_tetris_sidebar.svg",
    "figures/Figure2_tetris_sidebar.pdf": DRAFT_DIR / "Figure2_tetris_sidebar.pdf",
    "figures/Figure2_tetris_sidebar.png": DRAFT_DIR / "Figure2_tetris_sidebar.png",
    "FIGURE2_COMPACT_NOTES.md": DRAFT_DIR / "Figure2_tetris_README.md",
    "figures/Figure2_fiddle_with_fields.svg": DRAFT_DIR / "Figure2_fiddle_with_fields.svg",
    "figures/Figure2_fiddle_with_fields.pdf": DRAFT_DIR / "Figure2_fiddle_with_fields.pdf",
    "figures/Figure2_fiddle_with_fields.png": DRAFT_DIR / "Figure2_fiddle_with_fields.png",
    "figures/c15_temporal_s123_atlas.png": (
        DRAFT_DIR / "figure1_candidates" / "full_seed_atlases" /
        "c15_temporal_s123.png"),
    "figures/c15_constant_s123_atlas.png": (
        DRAFT_DIR / "figure1_candidates" / "full_seed_atlases" /
        "c15_constant_s123.png"),
    "FIGURE2_LAYOUT_NOTES.md": DRAFT_DIR / "Figure2_fiddle_with_fields_README.md",
    "figures/Figure2_story.svg": DRAFT_DIR / "Figure2_story.svg",
    "figures/Figure2_story.pdf": DRAFT_DIR / "Figure2_story.pdf",
    "figures/Figure2_story.png": DRAFT_DIR / "Figure2_story.png",
    "figures/Figure2_behavior_candidate_stats.svg": DRAFT_DIR / "Figure2_behavior_candidate_stats.svg",
    "figures/Figure2_behavior_candidate_stats.pdf": DRAFT_DIR / "Figure2_behavior_candidate_stats.pdf",
    "figures/Figure2_behavior_candidate_stats.png": DRAFT_DIR / "Figure2_behavior_candidate_stats.png",
    "FIGURE2_STORY_NOTES.md": DRAFT_DIR / "Figure2_story_README.md",
    "FIGURE2_STORY_SOURCES.json": DRAFT_DIR / "Figure2_story_manifest.json",
    "FIGURE2_INTRINSIC_REWARD_NOTES.md": (
        RESULTS_DIR / "cosyne_2027_intrinsic_reward_course/README.md"),
    "FIGURE2_INTRINSIC_REWARD_SOURCES.json": (
        RESULTS_DIR / "cosyne_2027_intrinsic_reward_course/render_manifest.json"),
    "FIGURE2_INTRINSIC_REWARD_RUNS.json": (
        RESULTS_DIR / "cosyne_2027_intrinsic_reward_course/collection_manifest.json"),
    "data/Figure2_intrinsic_reward_arm_summaries.csv": (
        RESULTS_DIR / "cosyne_2027_intrinsic_reward_course/arm_summaries.csv"),
    "data/Figure2_intrinsic_reward_by_seed.csv": (
        RESULTS_DIR / "cosyne_2027_intrinsic_reward_course/interpolated_by_seed.csv"),
    "figures/01b_online_intrinsic_reward_course.svg": (
        RESULTS_DIR / "cosyne_2027_intrinsic_reward_course/01b_online_intrinsic_reward_course.svg"),
    "figures/01b_online_intrinsic_reward_course.pdf": (
        RESULTS_DIR / "cosyne_2027_intrinsic_reward_course/01b_online_intrinsic_reward_course.pdf"),
    "figures/01b_online_intrinsic_reward_course.png": (
        RESULTS_DIR / "cosyne_2027_intrinsic_reward_course/01b_online_intrinsic_reward_course.png"),
    "data/fresh_behavior_decoder1_absolute.csv": (
        DRAFT_DIR / "data/fresh_behavior_decoder1_absolute.csv"),
    "data/fresh_behavior_training_seeds.csv": (
        DRAFT_DIR / "data/fresh_behavior_training_seeds.csv"),
    "data/Figure2_behavior_dg_bits_per_step.csv": (
        DRAFT_DIR / "figure2_candidates/08_focused_canvas/fresh_behavior_dg_bits_per_step.csv"),
    "data/Figure2_behavior_dg_reliability.csv": (
        DRAFT_DIR / "data/fresh_behavior_dg_reliability.csv"),
    "data/Figure2_behavior_geometry_by_model.csv": (
        DRAFT_DIR / "figure2_candidates/08_focused_canvas/behavior_geometry_by_model.csv"),
    "data/Figure2_behavior_geometry_provenance.json": (
        DRAFT_DIR / "figure2_candidates/08_focused_canvas/behavior_geometry_manifest.json"),
    "data/Figure2_trajectory_by_snapshot.csv": (
        RESULTS_DIR / "cosyne_2027_checkpoint_fields" / "graph_trajectory_summary" /
        "trajectory_by_snapshot.csv"),
    "figures/trajectory_examples/NG_s99.png": (
        RESULTS_DIR / "cosyne_2027_checkpoint_fields" / "selected_figures" /
        "trajectories/COS27_C01_TEMPORAL_S99.png"),
    "figures/trajectory_examples/UG_s99.png": (
        RESULTS_DIR / "cosyne_2027_checkpoint_fields" / "selected_figures" /
        "trajectories/COS27_C05_TEMPORAL_TEMPORAL_S99.png"),
    "figures/trajectory_examples/UG_s123.png": (
        DRAFT_DIR / "figure2_candidates/UG_s123_100m_trajectory.png"),
    "figures/trajectory_examples/UG_s123_tile.png": (
        DRAFT_DIR / "figure2_candidates/UG_s123_100m_trajectory_tile.png"),
    "figures/trajectory_examples/FG_s17.png": (
        RESULTS_DIR / "cosyne_2027_checkpoint_fields" / "selected_figures" /
        "trajectories/COS27_C15_TEMPORAL_S17.png"),
}
for metric in ("path_efficiency", "visited_cell_fraction",
               "stationary_step_fraction", "mean_physical_step_distance"):
    for extension in ("svg", "pdf", "png"):
        filename = f"Figure2_behavior_{metric}.{extension}"
        LAYOUT_TRIAL_ASSETS[f"figures/{filename}"] = DRAFT_DIR / filename

CURRENT_SVG_SOURCES = {
    "architecture_uploaded.svg": DRAFT_DIR / "figures" / "architecture_uploaded.svg",
    "results_sections.svg": DRAFT_DIR / "figures" / "results_sections.svg",
    "online_spatial_course.svg": RESULTS_DIR / "cosyne_2027_compact_figures" / "01_online_spatial_course.svg",
    "c01_peak_density.svg": RESULTS_DIR / "cosyne_2027_checkpoint_fields" / "common_panel_analysis" / "figures" / "c01_seed99_common_peak_density.svg",
    "encoder_paired_decoder_1.svg": RESULTS_DIR / "cosyne_2027_temporal_credit_layers" / "encoder_paired_decoder_1.svg",
    "behavior_expression_fresh.svg": DRAFT_DIR / "figures" / "behavior_expression_fresh.svg",
    "encoder_credit_deltas.svg": RESULTS_DIR / "cosyne_2027_compact_figures" / "03_encoder_credit_deltas.svg",
    "worker_coverage.svg": RESULTS_DIR / "cosyne_2027_compact_figures" / "04_worker_coverage_values_and_delta.svg",
}


def export_pdf(svg_path: Path, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["inkscape", str(svg_path), "--export-type=pdf", f"--export-filename={pdf_path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not pdf_path.is_file():
        raise RuntimeError(f"Could not export {svg_path}: {result.stderr.strip()}")


def write_archive(assets: dict[str, Path], path: Path) -> None:
    """Write a stable archive after checking every referenced source file."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for archive_name, source_path in sorted(assets.items()):
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            info = zipfile.ZipInfo(archive_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, source_path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)


def check_included_figures(assets: dict[str, Path]) -> None:
    included = set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",
                              (DRAFT_DIR / "main.tex").read_text()))
    missing = included.difference(assets)
    if missing:
        raise RuntimeError(f"main.tex references missing files: {sorted(missing)}")


def build_compact() -> None:
    """Package current figures, active layout trial, and submission notes."""
    pdf_names = ("architecture.pdf", "results_sections.pdf")
    svg_names = ("architecture_uploaded.svg", "results_sections.svg")
    with tempfile.TemporaryDirectory(prefix=".cosyne_compact_", dir=DRAFT_DIR) as temp_name:
        temp_dir = Path(temp_name)
        assets = {
            name: DRAFT_DIR / name for name in
            ("main.tex", "README.md", "OPEN_QUESTIONS.md", "PANEL_PROVENANCE.json")
        }
        assets.update({f"figures/{name}": DRAFT_DIR / "figures" / name
                       for name in pdf_names + svg_names})
        assets.update(LAYOUT_TRIAL_ASSETS)
        index = temp_dir / "FIGURE_OPTIONS.md"
        index.write_text("# Figures in this compact Overleaf project\n\n"
                         "The two PDFs below are used by `main.tex`; each has an SVG source.\n\n"
                         + "\n".join(f"- `figures/{name}`" for name in pdf_names)
                         + "\n\nThe compact editable Figure 2 layout trial is at "
                         "`figures/Figure2_tetris.svg`, with a PDF and preview "
                         "beside it. A tighter option with two empty side slots "
                         "is `figures/Figure2_tetris_sidebar.pdf`. "
                         "Read `FIGURE2_COMPACT_NOTES.md` and the "
                         "longer `FIGURE2_LAYOUT_NOTES.md` for its caption. "
                         "A story layout with trajectories and swappable behavior "
                         "statistics is `figures/Figure2_story.pdf`; see "
                         "`FIGURE2_STORY_NOTES.md`. The earlier taller trial is "
                         "also included. A matching five-seed mean intrinsic-"
                         "reward training course is available at "
                         "`figures/01b_online_intrinsic_reward_course.pdf`; "
                         "see `FIGURE2_INTRINSIC_REWARD_NOTES.md`.\n\nThe expanded candidate "
                         "library is in the repository's full archive.\n")
        assets[index.name] = index
        check_included_figures(assets)
        staged = temp_dir / ARCHIVE_PATH.name
        write_archive(assets, staged)
        staged.replace(ARCHIVE_PATH)
    print(f"Packaged {len(pdf_names)} current PDFs and their SVG sources: {ARCHIVE_PATH}")


def build_full() -> None:
    current_pdfs = sorted((DRAFT_DIR / "figures").glob("*.pdf"))
    for svg_name in CURRENT_SVG_SOURCES:
        expected_pdf = DRAFT_DIR / "figures" / f"{Path(svg_name).stem.removesuffix('_uploaded')}.pdf"
        if not expected_pdf.is_file():
            raise FileNotFoundError(expected_pdf)

    candidate_svgs = sorted(RESULTS_DIR.glob("cosyne_2027_*/**/*.svg"))
    raster_only = sorted(
        path for path in RESULTS_DIR.glob("cosyne_2027_*/**/*.png")
        if not path.with_suffix(".svg").exists()
    )
    if not candidate_svgs:
        raise RuntimeError("No COSYNE candidate SVGs were found")

    # Keep staging on the same filesystem so the final ZIP replacement is atomic.
    with tempfile.TemporaryDirectory(prefix=".cosyne_overleaf_", dir=DRAFT_DIR) as temp_name:
        temp_dir = Path(temp_name)
        assets: dict[str, Path] = {
            "main.tex": DRAFT_DIR / "main.tex",
            "README.md": DRAFT_DIR / "README.md",
            "OPEN_QUESTIONS.md": DRAFT_DIR / "OPEN_QUESTIONS.md",
            "render_submission_panels.py": DRAFT_DIR / "render_submission_panels.py",
            "PANEL_PROVENANCE.json": DRAFT_DIR / "PANEL_PROVENANCE.json",
            "references.bib": REPO_ROOT / "08_literature" / "references.bib",
        }
        for data_path in sorted((DRAFT_DIR / "data").glob("*")):
            if data_path.is_file():
                assets[f"data/{data_path.name}"] = data_path
        for pdf_path in current_pdfs:
            assets[f"figures/{pdf_path.name}"] = pdf_path
        for png_path in sorted((DRAFT_DIR / "figures").glob("*.png")):
            assets[f"figures/{png_path.name}"] = png_path
        for svg_name, source_path in CURRENT_SVG_SOURCES.items():
            assets[f"figures/{svg_name}"] = source_path
        assets.update(LAYOUT_TRIAL_ASSETS)

        groups: dict[str, list[str]] = {}
        for source_path in candidate_svgs:
            relative = source_path.relative_to(RESULTS_DIR)
            archive_svg = Path("figures/candidates") / relative
            archive_pdf = archive_svg.with_suffix(".pdf")
            exported_pdf = temp_dir / archive_pdf
            export_pdf(source_path, exported_pdf)
            assets[archive_svg.as_posix()] = source_path
            assets[archive_pdf.as_posix()] = exported_pdf
            groups.setdefault(relative.parts[0], []).append(archive_pdf.as_posix())
        for source_path in raster_only:
            relative = source_path.relative_to(RESULTS_DIR)
            archive_png = Path("figures/candidates") / relative
            assets[archive_png.as_posix()] = source_path
            groups.setdefault(relative.parts[0], []).append(archive_png.as_posix())

        check_included_figures(assets)

        index_lines = [
            "# Figure options for the COSYNE draft",
            "",
            "The PDFs in `figures/` are used by `main.tex`. Each has an SVG source",
            "beside it. Additional options are in `figures/candidates/`, organized",
            "by the source analysis folder. Every candidate PDF has a matching SVG;",
            "the PNG-only files are place-field, graph, and trajectory examples.",
            "Change an `\\includegraphics` path in `main.tex` to try an option.",
            "Candidates include exploratory and historical plots; check the linked",
            "analysis before making a scientific claim from a replacement panel.",
            "",
            "## Current figure assets",
            "",
        ]
        index_lines.extend(f"- `figures/{pdf_path.name}`" for pdf_path in current_pdfs)
        index_lines.extend(["", "## Compact Figure 2 layout trial", "",
                            "- `figures/Figure2_tetris.pdf`",
                            "- `figures/Figure2_tetris.svg`",
                            "- `figures/Figure2_tetris_sidebar.pdf`",
                            "- `figures/Figure2_tetris_sidebar.svg`",
                            "- `FIGURE2_COMPACT_NOTES.md`",
                            "", "## Figure 2 story layout and behavior-stat alternatives", "",
                            "- `figures/Figure2_story.pdf`",
                            "- `figures/Figure2_story.svg`",
                            "- `figures/Figure2_behavior_candidate_stats.pdf`",
                            "- `FIGURE2_STORY_NOTES.md`",
                            "", "## Mean intrinsic reward course", "",
                            "- `figures/01b_online_intrinsic_reward_course.pdf`",
                            "- `figures/01b_online_intrinsic_reward_course.svg`",
                            "- `FIGURE2_INTRINSIC_REWARD_NOTES.md`",
                            "", "## Earlier Figure 2 layout trial", "",
                            "- `figures/Figure2_fiddle_with_fields.pdf`",
                            "- `figures/Figure2_fiddle_with_fields.svg`",
                            "- `FIGURE2_LAYOUT_NOTES.md`"])
        for group, pdf_paths in sorted(groups.items()):
            index_lines.extend(["", f"## {group} ({len(pdf_paths)} plots)", ""])
            index_lines.extend(f"- `{path}`" for path in pdf_paths)
        index_path = temp_dir / "FIGURE_OPTIONS.md"
        index_path.write_text("\n".join(index_lines) + "\n")
        assets["FIGURE_OPTIONS.md"] = index_path

        staged_zip = temp_dir / FULL_ARCHIVE_PATH.name
        write_archive(assets, staged_zip)
        staged_zip.replace(FULL_ARCHIVE_PATH)
        print(f"Packaged {len(current_pdfs)} current PDFs, {len(candidate_svgs)} "
              f"candidate PDF/SVG pairs, {len(raster_only)} PNG-only examples, "
              f"and {len(CURRENT_SVG_SOURCES)} current SVGs")
        print(FULL_ARCHIVE_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="write a separate archive with all candidate figures")
    args = parser.parse_args()
    if args.full:
        build_full()
    else:
        build_compact()
