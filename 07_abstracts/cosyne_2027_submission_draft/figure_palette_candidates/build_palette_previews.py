"""Recolor the current COSYNE figures for semantic palette comparison.

This script changes only enumerated vector colors. It preserves the figure
geometry, plotted values, embedded images, Viridis trajectory scale, and
place-field images. Each preview is derived afresh from the checked-in source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
DRAFT = HERE.parent
SOURCES = {
    "figure1": DRAFT / "figure1_candidates/06_architecture_v3_goal_gate_header.svg",
    "figure2": DRAFT / "Figure2_story_icons.svg",
}
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
EXPECTED_WIDTH = 2250


@dataclass(frozen=True)
class Palette:
    name: str
    description: str
    ng: str
    ug: str
    fg: str
    push: str
    pull: str
    dg_blue: str
    dg_red: str
    dg_yellow: str
    dg_fourth: str
    pull_dashed: bool = False


PALETTES = (
    Palette(
        name="slate_ng",
        description="Neutral NG; red push and blue pull; violet, ochre, and teal DG identities",
        ng="#586674", ug="#D55E00", fg="#009E73",
        push="#9D3349", pull="#175E91",
        dg_blue="#735DA8", dg_red="#C7922F", dg_yellow="#3E9C9C",
        dg_fourth="#C477A5",
    ),
    Palette(
        name="slate_ng_violet_rose",
        description="Option A with the two highlighted DG identities reassigned to violet and rose",
        ng="#586674", ug="#D55E00", fg="#009E73",
        push="#9D3349", pull="#175E91",
        dg_blue="#C7922F", dg_red="#735DA8", dg_yellow="#C477A5",
        dg_fourth="#3E9C9C",
    ),
    Palette(
        name="slate_ng_tuned_dg",
        description="Violet and rose highlighted DG identities; quieter taupe and cyan for the other two",
        ng="#586674", ug="#D55E00", fg="#009E73",
        push="#9D3349", pull="#175E91",
        dg_blue="#81705C", dg_red="#735DA8", dg_yellow="#C477A5",
        dg_fourth="#4D9EAF",
    ),
    Palette(
        name="purple_pull",
        description="NG/UG/FG stay categorical; umber push and purple pull; cyan, rose, and ochre DG identities",
        ng="#276FA6", ug="#C85F24", fg="#198E71",
        push="#8C5231", pull="#6E52A3",
        dg_blue="#3AA9B4", dg_red="#C2536F", dg_yellow="#C7A83F",
        dg_fourth="#687F55",
    ),
    Palette(
        name="role_neutral",
        description="Project color standard: architecture owns hue; neutral solid push, dashed pull; ochre, violet, and rose DG identities",
        ng="#0072B2", ug="#D55E00", fg="#009E73",
        push="#424B53", pull="#424B53",
        dg_blue="#B58C22", dg_red="#8B6CA8", dg_yellow="#BD5C89",
        dg_fourth="#75838C", pull_dashed=True,
    ),
    Palette(
        name="rgb_magenta_teal",
        description="Opposing red, green, and blue DG IDs; magenta push and teal pull",
        ng="#586674", ug="#D55E00", fg="#009E73",
        push="#A5387B", pull="#0B8293",
        dg_blue="#315BC4", dg_red="#D8493B", dg_yellow="#359B51",
        dg_fourth="#B29924",
    ),
    Palette(
        name="rgb_orange_violet",
        description="Opposing red, green, and blue DG IDs; orange push and violet pull",
        ng="#586674", ug="#D55E00", fg="#009E73",
        push="#BF5828", pull="#6952A6",
        dg_blue="#2F67C8", dg_red="#CB3C54", dg_yellow="#2D9F5B",
        dg_fourth="#25A4A4",
    ),
    Palette(
        name="cmy_red_blue",
        description="Cyan, magenta, and yellow DG IDs; retain A's red push and blue pull",
        ng="#586674", ug="#D55E00", fg="#009E73",
        push="#9D3349", pull="#175E91",
        dg_blue="#26A7B8", dg_red="#BE4AA1", dg_yellow="#B8A12E",
        dg_fourth="#3E9A5C",
    ),
    Palette(
        name="priority_rgb_magenta_teal",
        description="Slate NG, copper UG, lilac FG; opposing blue, red, green DG IDs; magenta push and teal pull",
        ng="#59636B", ug="#B9793B", fg="#9C6EC3",
        push="#A5387B", pull="#0B8293",
        dg_blue="#315BC4", dg_red="#D8493B", dg_yellow="#359B51",
        dg_fourth="#B29924",
    ),
    Palette(
        name="priority_rgb_orange_violet",
        description="Slate NG, olive UG, rose FG; opposing blue, red, green DG IDs; orange push and violet pull",
        ng="#59636B", ug="#999A35", fg="#C567A7",
        push="#BF5828", pull="#6952A6",
        dg_blue="#2F67C8", dg_red="#CB3C54", dg_yellow="#2D9F5B",
        dg_fourth="#25A4A4",
    ),
    Palette(
        name="priority_rgb_amber_violet",
        description="Slate NG, olive UG, rose FG; opposing blue, red, green DG IDs; amber push and violet pull",
        ng="#59636B", ug="#8E9D36", fg="#C577AF",
        push="#AD6B19", pull="#7046AD",
        dg_blue="#265FC7", dg_red="#D43E3E", dg_yellow="#299C5F",
        dg_fourth="#27A4B3",
    ),
)
PALETTE_LABELS = {
    "slate_ng": "A · Slate NG",
    "slate_ng_violet_rose": "A2 · Violet + rose DG",
    "slate_ng_tuned_dg": "A3 · Tuned remaining DG colors",
    "purple_pull": "B · Purple pull",
    "role_neutral": "C · Neutral mechanisms",
    "rgb_magenta_teal": "D · RGB DG / magenta–teal",
    "rgb_orange_violet": "E · RGB DG / orange–violet",
    "cmy_red_blue": "F · CMY DG / red–blue",
    "priority_rgb_magenta_teal": "G · Copper/lilac groups / magenta–teal",
    "priority_rgb_orange_violet": "H · Olive/rose groups / orange–violet",
    "priority_rgb_amber_violet": "I · Olive/rose groups / amber–violet",
}
PALETTE_KEY_LABELS = {
    "slate_ng": "A · Slate NG",
    "slate_ng_violet_rose": "A2 · Violet/rose",
    "slate_ng_tuned_dg": "A3 · Tuned DG",
    "purple_pull": "B · Purple pull",
    "role_neutral": "C · Neutral",
    "rgb_magenta_teal": "D · RGB/M–T",
    "rgb_orange_violet": "E · RGB/O–V",
    "cmy_red_blue": "F · CMY/R–B",
    "priority_rgb_magenta_teal": "G · RGB/M–T",
    "priority_rgb_orange_violet": "H · RGB/O–V",
    "priority_rgb_amber_violet": "I · RGB/A–V",
}
REVIEW_PALETTE_NAMES = (
    "slate_ng_violet_rose", "slate_ng_tuned_dg",
    "rgb_magenta_teal", "rgb_orange_violet", "cmy_red_blue",
    "priority_rgb_magenta_teal", "priority_rgb_orange_violet",
    "priority_rgb_amber_violet",
)
SELECTED_PALETTE_NAME = "priority_rgb_orange_violet"
BASELINE = Palette(
    name="current", description="Current source palette",
    ng="#0072B2", ug="#D55E00", fg="#009E73",
    push="#AE453E", pull="#236D9D",
    dg_blue="#4364A6", dg_red="#A73E30", dg_yellow="#C3B058",
    dg_fourth="#8864A8",
)


def tint(color: str, white_fraction: float) -> str:
    """Return a lighter fill for a process box or translucent unit row."""
    rgb = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
    return "#" + "".join(
        f"{round(channel * (1 - white_fraction) + 255 * white_fraction):02X}"
        for channel in rgb
    )


def replacement_map(palette: Palette) -> dict[str, str]:
    """Map the exact source colors by meaning, not by approximate hue."""
    groups = (
        (palette.ng, ("#0072B2",)),
        (palette.ug, ("#D55E00",)),
        (palette.fg, ("#009E73",)),
        (palette.push, ("#AE453E", "#933333", "#861B1B")),
        (tint(palette.push, 0.52), ("#C58C8C",)),
        (palette.pull, ("#236D9D", "#136199", "#136299", "#16639A")),
        (tint(palette.pull, 0.64), ("#ABC9E0",)),
        (palette.dg_blue, ("#4364A6", "#4264A6", "#4364A5", "#4164A5", "#3178C6")),
        (palette.dg_red, ("#A73E30", "#A83D2F", "#A73E2F", "#A83C2F")),
        (tint(palette.dg_red, 0.32), ("#ED805A",)),
        (palette.dg_yellow, ("#C3B058", "#C3B158")),
        (palette.dg_fourth, ("#8864A8",)),
    )
    mapping = {source.lower(): target for target, sources in groups
               for source in sources}
    if len(mapping) != sum(len(sources) for _, sources in groups):
        raise ValueError("A source color was assigned to two semantic roles")
    return mapping


def recolor(source: Path, destination: Path, palette: Palette) -> dict[str, int]:
    content = source.read_text()
    if palette.pull_dashed:
        if source == SOURCES["figure1"]:
            content, count = re.subn(r"\.pull \{", ".pull { stroke-dasharray: 6 4;", content)
            if count != 1:
                raise ValueError("Could not style the Figure 1 pull path")
            content, count = re.subn(
                r'(<path\s+style="[^"]*stroke:#16639a;[^"]*?)stroke-dasharray:none',
                r'\1stroke-dasharray:4,3', content,
            )
            if count != 1:
                raise ValueError("Could not dash the Figure 1 architecture pull arrow")
            content, count = re.subn(
                r'(<rect\s+style="[^"]*stroke:#136299;[^"]*?)stroke-dasharray:none',
                r'\1stroke-dasharray:4,2.5', content,
            )
            if count != 1:
                raise ValueError("Could not dash the Figure 1 pull legend key")
            content, count = re.subn(
                r'(<path\b[^>]*?stroke="#236d9d")',
                r'\1 stroke-dasharray="2 1.3"', content,
            )
            if count != 1:
                raise ValueError("Could not dash the Figure 1 goal-gate pull arrow")
        else:
            content, count = re.subn(
                r'(<path\b[^>]*?stroke="#16639a" stroke-width="1\.2")',
                r'\1 stroke-dasharray="2.1 1.6"', content,
            )
            if count != 2:
                raise ValueError(f"Expected two Figure 2 pull arrows, got {count}")
    mapping = replacement_map(palette)
    counts = {key: 0 for key in mapping}

    def change(match: re.Match[str]) -> str:
        key = match.group(0).lower()
        if key in mapping:
            counts[key] += 1
            return mapping[key]
        return match.group(0)

    result = re.sub(r"#[0-9a-fA-F]{6}\b", change, content)
    required = ("#0072b2", "#d55e00", "#009e73", "#a73e30", "#4364a6",
                "#c3b058", "#933333", "#236d9d")
    missing = [key for key in required if counts[key] == 0]
    if missing:
        raise ValueError(f"{source.name} lacks expected semantic colors: {missing}")
    destination.write_text(result)
    return {key: value for key, value in counts.items() if value}


def export(svg: Path) -> None:
    subprocess.run(["inkscape", str(svg), "--export-type=pdf",
                    f"--export-filename={svg.with_suffix('.pdf')}"], check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run(["inkscape", str(svg), f"--export-width={EXPECTED_WIDTH}",
                    f"--export-filename={svg.with_suffix('.png')}"], check=True,
                   stdout=subprocess.DEVNULL)


def comparison_sheet(figure: str, group: str) -> None:
    if not FONT.is_file():
        raise FileNotFoundError(f"Scalable font required: {FONT}")
    variants = [("Current", SOURCES[figure].with_suffix(".png"))]
    selected_names = {
        "all": {palette.name for palette in PALETTES[:5]},
        "A_refinements": {palette.name for palette in PALETTES[:3]},
        "opponent_dg": {"slate_ng_violet_rose", "slate_ng_tuned_dg", "rgb_magenta_teal",
                        "rgb_orange_violet", "cmy_red_blue"},
        "role_first": {"priority_rgb_magenta_teal", "priority_rgb_orange_violet",
                       "priority_rgb_amber_violet"},
    }[group]
    selected = (palette for palette in PALETTES
                if palette.name in selected_names)
    variants.extend((PALETTE_LABELS[palette.name],
                     HERE / f"{palette.name}_{figure}.png")
                    for palette in selected)
    header = 75
    gap = 22
    loaded: list[tuple[str, Image.Image]] = []
    for label, path in variants:
        image = Image.open(path).convert("RGB")
        if image.width != EXPECTED_WIDTH:
            height = round(image.height * EXPECTED_WIDTH / image.width)
            image = image.resize((EXPECTED_WIDTH, height), Image.Resampling.LANCZOS)
        loaded.append((label, image))
    total_height = sum(header + image.height + gap for _, image in loaded) + gap
    sheet = Image.new("RGB", (EXPECTED_WIDTH, total_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(FONT), 55)
    top = gap
    for label, image in loaded:
        draw.text((35, top), label, fill="#1F2933", font=font)
        top += header
        sheet.paste(image, (0, top))
        top += image.height + gap
    suffix = "" if group == "all" else f"_{group}"
    sheet.save(HERE / f"comparison_{figure}{suffix}.png", optimize=True)


def palette_key(selected_names: tuple[str, ...] | None = None,
                filename: str = "palette_key.png") -> None:
    """Make role assignments explicit beside the full-figure comparisons."""
    if not FONT.is_file():
        raise FileNotFoundError(f"Scalable font required: {FONT}")
    columns = (("ng", "NG"), ("ug", "UG"), ("fg", "FG"),
               ("push", "Push"), ("pull", "Pull"),
               ("dg_blue", "DG 1"), ("dg_red", "DG 2"),
               ("dg_yellow", "DG 3"), ("dg_fourth", "DG 4"))
    left, column_width, row_height = 330, 205, 145
    selected = [palette for palette in PALETTES
                if selected_names is None or palette.name in selected_names]
    sheet = Image.new("RGB", (left + len(columns) * column_width + 25,
                              110 + (1 + len(selected)) * row_height), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = ImageFont.truetype(str(FONT), 40)
    label_font = ImageFont.truetype(str(FONT), 34)
    code_font = ImageFont.truetype(str(FONT), 26)
    for index, (_, label) in enumerate(columns):
        draw.text((left + index * column_width + 10, 25), label,
                  font=label_font, fill="#23313B")
    choices = [("Current", BASELINE)] + [(PALETTE_KEY_LABELS[p.name], p)
                                       for p in selected]
    for row, (name, palette) in enumerate(choices):
        top = 100 + row * row_height
        draw.text((18, top + 20), name, font=title_font, fill="#23313B")
        for index, (attribute, _) in enumerate(columns):
            color = getattr(palette, attribute)
            x = left + index * column_width + 10
            draw.rounded_rectangle((x, top, x + 150, top + 74), radius=10,
                                   fill=color, outline="#23313B", width=2)
            if palette.pull_dashed and attribute in {"push", "pull"}:
                segments = ([(x + 20, x + 126)] if attribute == "push" else
                            [(x + start, x + start + 15) for start in (20, 46, 72, 98)])
                for begin, end in segments:
                    draw.line((begin, top + 37, end, top + 37), fill="white", width=6)
                draw.polygon(((x + 129, top + 37), (x + 116, top + 26),
                              (x + 116, top + 48)), fill="white")
            draw.text((x, top + 83), color.upper(), font=code_font,
                      fill="#23313B")
    sheet.save(HERE / filename, optimize=True)


def paired_pdf(name: str) -> None:
    """Combine the unchanged-size vector figures into a flip-through PDF."""
    inputs = [HERE / f"{name}_{figure}.pdf" for figure in SOURCES]
    output = HERE / f"review_{name}.pdf"
    if output.exists() and output.stat().st_mtime_ns >= max(
            path.stat().st_mtime_ns for path in inputs):
        return
    subprocess.run(["pdfunite", *(str(path) for path in inputs), str(output)],
                   check=True, stdout=subprocess.DEVNULL)


def publish_selected_figures() -> dict[str, dict[str, str]]:
    """Expose the selected vector pair under stable submission-draft names."""
    published = {}
    for figure in SOURCES:
        outputs = {}
        for suffix in ("svg", "pdf"):
            source = HERE / f"{SELECTED_PALETTE_NAME}_{figure}.{suffix}"
            destination = DRAFT / f"{figure.capitalize()}.{suffix}"
            content = source.read_bytes()
            if not destination.exists() or destination.read_bytes() != content:
                destination.write_bytes(content)
            outputs[suffix] = str(destination.relative_to(DRAFT))
        published[figure] = outputs
    return published


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    if SELECTED_PALETTE_NAME not in {palette.name for palette in PALETTES}:
        raise ValueError(f"Unknown selected palette: {SELECTED_PALETTE_NAME}")
    provenance = {
        "schema": "cosyne27-color-palette-preview/v1",
        "selected_palette": SELECTED_PALETTE_NAME,
        "sources": {name: {"path": str(path.relative_to(DRAFT)),
                           "sha256": sha256(path.read_bytes()).hexdigest()}
                    for name, path in SOURCES.items()},
        "scope": "Exact vector color substitutions only; geometry, data, Viridis trajectory colors, embedded images, and place fields unchanged",
        "palettes": {},
    }
    for palette in PALETTES:
        records = {}
        for figure, source in SOURCES.items():
            target = HERE / f"{palette.name}_{figure}.svg"
            previous_svg = target.read_bytes() if target.exists() else None
            records[figure] = {"replacements": recolor(source, target, palette),
                               "svg_sha256": sha256(target.read_bytes()).hexdigest()}
            if (previous_svg != target.read_bytes() or
                    not target.with_suffix(".pdf").exists() or
                    not target.with_suffix(".png").exists()):
                export(target)
        provenance["palettes"][palette.name] = {"colors": asdict(palette),
                                                 "figures": records}
    for name in REVIEW_PALETTE_NAMES:
        paired_pdf(name)
    provenance["published_figures"] = publish_selected_figures()
    for figure in SOURCES:
        for group in ("all", "A_refinements", "opponent_dg", "role_first"):
            comparison_sheet(figure, group)
    palette_key()
    palette_key(("priority_rgb_magenta_teal", "priority_rgb_orange_violet",
                 "priority_rgb_amber_violet"), "palette_key_role_first.png")
    (HERE / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
