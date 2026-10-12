"""Build the selected Overleaf project or its expanded figure library.

The default compact archive contains exactly the current figure sources and
PDFs. ``--full`` writes a separate archive with candidate COSYNE plots; it
never replaces the current upload package.
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
    """Package only current PDF figures, their SVG sources, and submission notes."""
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
        index = temp_dir / "FIGURE_OPTIONS.md"
        index.write_text("# Figures in this compact Overleaf project\n\n"
                         "The two PDFs below are used by `main.tex`; each has an SVG source.\n\n"
                         + "\n".join(f"- `figures/{name}`" for name in pdf_names)
                         + "\n\nThe expanded candidate library is in the repository's full archive.\n")
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
