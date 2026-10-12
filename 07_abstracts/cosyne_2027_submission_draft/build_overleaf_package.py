"""Build a self-contained Overleaf project with the COSYNE figure library.

The figures used by main.tex stay at their existing paths. Other COSYNE SVGs
are included under figures/candidates/ with matching PDFs, so a candidate can
be selected by changing only an \includegraphics path in Overleaf.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import zipfile
from pathlib import Path


DRAFT_DIR = Path(__file__).resolve().parent
REPO_ROOT = DRAFT_DIR.parents[1]
RESULTS_DIR = REPO_ROOT / "06_experiments" / "results"
ARCHIVE_PATH = DRAFT_DIR / "cosyne_2027_overleaf_sources.zip"

CURRENT_SVG_SOURCES = {
    "architecture_uploaded.svg": DRAFT_DIR / "figures" / "architecture_uploaded.svg",
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


def build() -> None:
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
            "references.bib": REPO_ROOT / "08_literature" / "references.bib",
        }
        for pdf_path in current_pdfs:
            assets[f"figures/{pdf_path.name}"] = pdf_path
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

        included_figures = set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",
                                          (DRAFT_DIR / "main.tex").read_text()))
        missing_figures = included_figures.difference(assets)
        if missing_figures:
            raise RuntimeError(f"main.tex references missing files: {sorted(missing_figures)}")

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

        staged_zip = temp_dir / ARCHIVE_PATH.name
        with zipfile.ZipFile(staged_zip, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=9) as archive:
            for archive_name, source_path in sorted(assets.items()):
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                archive.write(source_path, archive_name)
        staged_zip.replace(ARCHIVE_PATH)
        print(f"Packaged {len(current_pdfs)} current PDFs, {len(candidate_svgs)} "
              f"candidate PDF/SVG pairs, {len(raster_only)} PNG-only examples, "
              f"and {len(CURRENT_SVG_SOURCES)} current SVGs")
        print(ARCHIVE_PATH)


if __name__ == "__main__":
    build()
