"""Fill the right column of the supplied A0 poster from pinned local results.

The source SVG is an explicit input. Only identified right-column objects are
removed; the header and every retained source subtree are preserved. New plots
are native SVG groups with editable text and unique IDs. Run with SF_git Python.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import xml.etree.ElementTree as ET

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'intrmotiv-poster-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont, FontProperties
from matplotlib.lines import Line2D
from matplotlib.textpath import TextToPath
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# The canonical survey renderer imports the root study package. Keep the
# artifact adapter runnable directly from any current working directory.
sys.path.insert(0, str(ROOT))
from render_all_run_spatial_scatter import draw_scatter, FAMILY_COLORS

RESULTS = ROOT / '06_experiments/results/A0_poster_analysis_20260926'
TABLE = RESULTS / 'flat_goal_comparison/matched_terminal_per_run.csv'
SCATTER = ROOT / '06_experiments/data/poster_missing_analyses_20260926/cross_run_scatter/all_run_metrics.csv'
SVG = 'http://www.w3.org/2000/svg'
INK = 'http://www.inkscape.org/namespaces/inkscape'
SOD = 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd'
COLORS = {'C01': '#5E5E5E', 'C05': '#009E73', 'C15': '#D55E00'}
SEEDS = (8, 99, 123)
FRAME = 100_040_704
PT_MM = 25.4 / 72
PLOT_FONT_PT = 30
BODY_FONT_PT = 40
HEADING_FONT_PT = 48
# Capacity is fixed in the main survey, so shape can reinforce study family.
FAMILY_MARKERS = {'CPD': 'o', 'DGP': 's', 'DGC': '^', 'CPU cadence': 'D',
                  'Navigation8': 'P', 'Source credit': 'X'}
# Exact objects found in the reviewed source. No broad coordinate deletion.
REMOVE = {
    'text33-0-6-9-3-6-62', 'text5077-0-8', 'text5079-5-9',
    'text5079-5-9-2', 'text5079-5-9-5', 'text96',
    'text5079-5-9-3', 'text5077-8-3-3',
    'text131-6-1-1-7-5-2-3-3-1-1-7', 'figure_1-90', 'figure_1',
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style() -> str:
    font = Path(findfont('DejaVu Sans', fallback_to_default=False))
    if not font.is_file() or font.suffix.lower() not in {'.ttf', '.otf'}:
        raise RuntimeError(f'No verified scalable font: {font}')
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': PLOT_FONT_PT, 'axes.labelsize': PLOT_FONT_PT,
        'axes.titlesize': PLOT_FONT_PT, 'xtick.labelsize': PLOT_FONT_PT, 'ytick.labelsize': PLOT_FONT_PT,
        'legend.fontsize': PLOT_FONT_PT, 'svg.fonttype': 'none', 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': 1.2, 'svg.hashsalt': 'intrmotiv-a0-20260929',
    })
    return str(font)


def dots(ax, data: pd.DataFrame, key: str, conditions: tuple[str, ...], factor=1.) -> None:
    for i, c in enumerate(conditions):
        values = data.loc[data.condition == c].sort_values('seed')[key].to_numpy(float) * factor
        ax.scatter(i + np.array([-.12, 0, .12]), values, color=COLORS[c], s=56, zorder=3)
        ax.plot([i-.19, i+.19], [values.mean()]*2, color='black', lw=2.2, zorder=4)
    ax.set_xticks(range(len(conditions)), conditions)
    ax.set_xlim(-.5, len(conditions)-.5)
    ax.grid(axis='y', color='#e5e5e5', zorder=0)


def save(fig, output: Path) -> None:
    fig.savefig(output, metadata={'Date': None})
    plt.close(fig)


def render_plots(data: pd.DataFrame, out: Path) -> tuple[list[Path], list[dict]]:
    # Large labels belong outside the small data area. Short two-line titles
    # replace repeated vertical labels while preserving quantities and units.
    fig, axes = plt.subplots(1, 4, figsize=(383 / 25.4, 3.0), layout='constrained')
    specs = [('coverage_auc_terminal', 'A  AUC\n(cells)', 1),
             ('unique_cells_terminal', 'B  Cells\n/ episode', 1),
             ('return_20_mobile', 'C  Return (%)\n20 decisions', 100),
             ('return_40_mobile', 'D  Return (%)\n40 decisions', 100)]
    for ax, (key, title, factor) in zip(axes.flat, specs):
        dots(ax, data, key, ('C01', 'C05', 'C15'), factor)
        ax.set(title=title)
        ax.set_ylim(0, 100 if factor == 100 or key.startswith('coverage') else 175)
        if factor == 100 or key.startswith('coverage'):
            ax.set_yticks([0, 50, 100])
        else:
            ax.set_yticks([0, 100])
    overview = out / 'exploration_and_returns.svg'
    save(fig, overview)

    fig, axes = plt.subplots(1, 2, figsize=(191.5 / 25.4, 3.0), layout='constrained')
    dots(axes[0], data, 'target_hit_lift_terminal', ('C05', 'C15'))
    axes[0].axhline(1, color='#777777', ls='--', lw=1.5)
    axes[0].set(title='A  Hit lift\nvs shuffled',
                ylim=(.8, 1.2), yticks=[.8, 1, 1.2])
    dots(axes[1], data, 'action_sensitivity_terminal', ('C05', 'C15'))
    axes[1].set(title='B  Goal change\nmean |Δlogit|',
                ylim=(0, .055), yticks=[0, .02, .04])
    control = out / 'goal_specificity.svg'
    save(fig, control)

    scatter, cohorts = render_survey(out)
    return [overview, control, scatter], cohorts


def capacity_audit(points: pd.DataFrame, out: Path) -> None:
    """Screen capacities separately and retain family-specific associations.

    This small view of the pinned survey changes no study or telemetry data.
    Constant/missing axes remain explicitly undefined. The audit includes
    protocols omitted from the main poster so candidate stories are traceable.
    """
    records = []
    pairs = [('spatial_information', 'prospective_success'),
             ('unique_peak_bins', 'prospective_success'),
             ('spatial_information', 'exploration_coverage'),
             ('spatial_information', 'executed_minus_shuffled')]
    for (protocol, capacity), group in points.groupby(['protocol', 'dg_units']):
        strata = [('ALL', group), *list(group.groupby('family'))]
        for family, rows in strata:
            for x, y in pairs:
                finite = rows.dropna(subset=[x, y])
                rho = (float(finite[x].corr(finite[y], method='spearman'))
                       if finite[x].nunique() > 1 and finite[y].nunique() > 1 else None)
                records.append({'protocol': protocol, 'dg_units': int(capacity),
                                'family': family, 'x': x, 'y': y, 'n': len(finite),
                                'conditions': finite.condition.nunique(), 'spearman_rho': rho,
                                'minimum_frames': finite.frames.min(),
                                'maximum_frames': finite.frames.max()})
    pd.DataFrame(records).to_csv(out / 'dg_capacity_audit.csv', index=False)


def render_survey(out: Path) -> tuple[Path, list[dict]]:
    """Show two metrics on the same eligible DG-16 online run cohort.

    DG 32/64 are screened separately rather than pooled into architecture
    comparisons. A/B repeat the same runs, with family shapes and colors.
    Correlations describe saved windows and are not causal effect estimates.
    """
    points = pd.read_csv(SCATTER)
    legacy = points[points.geometry_group == 'legacy_19x19']
    capacity_audit(legacy, out)
    online = legacy[(legacy.protocol == 'online_latest_saved_window') & (legacy.dg_units == 16)]
    definitions = [
        ('A', 'spatial_information', 'A  Spatial score'),
        ('B', 'unique_peak_bins', 'B  Peak diversity'),
    ]
    cohorts, displayed = [], []
    fig, axes = plt.subplots(1, 2, figsize=(383 / 25.4, 6.5), sharey=True)
    fig.subplots_adjust(left=.125, right=.985, top=.85, bottom=.405, wspace=.26)
    for ax, (panel, metric, title) in zip(axes.flat, definitions):
        outcome = 'prospective_success'
        usable = online.dropna(subset=['spatial_information', 'unique_peak_bins', outcome]).copy()
        if usable.run_name.duplicated().any():
            raise ValueError(f'Duplicate run in survey panel {panel}')
        rho = float(usable[metric].corr(usable[outcome], method='spearman'))
        cohorts.append({'panel': panel, 'protocol': str(usable.protocol.iloc[0]),
                        'dg_units': 16, 'x': metric, 'y': outcome, 'n': len(usable),
                        'conditions': int(usable.condition.nunique()),
                        'spearman_rho': rho, 'minimum_frames': int(usable.frames.min()),
                        'maximum_frames': int(usable.frames.max()),
                        'families': usable.groupby('family').size().to_dict()})
        usable['poster_panel'] = panel
        displayed.append(usable)
        draw_scatter(ax, usable, metric, outcome, title,
                     marker_column='family', markers=FAMILY_MARKERS)
        for collection in ax.collections:
            collection.set_sizes([65])
        ax.set(title=title, ylim=(0, 1), yticks=[0, .5, 1])
        if panel == 'A':
            ax.set(xlim=(0, .56), xticks=[0, .2, .4], xlabel='Spatial score', ylabel='Target hits / attempts')
        else:
            ax.set(xlim=(6.5, 16.5), xticks=[8, 12, 16], xlabel='Distinct peak bins', ylabel='')
        cohorts[-1]['within_family_rho'] = {
            family: float(group[metric].corr(group[outcome], method='spearman'))
            for family, group in usable.groupby('family')}
        ax.text(.04, .96, f'ρ = {rho:.2f}', transform=ax.transAxes, va='top', fontsize=PLOT_FONT_PT)
    families = sorted(set().union(*(set(frame.family) for frame in displayed)))
    legend_labels = {'CPD': 'CA3 feedback', 'DGP': 'DG policy', 'DGC': 'Goal interfaces',
                     'CPU cadence': 'Controllers', 'Navigation8': 'Algorithm screen'}
    handles = [Line2D([], [], marker=FAMILY_MARKERS[family], color=FAMILY_COLORS[family], linestyle='none',
                      markersize=8, label=legend_labels.get(family, family)) for family in families]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .025),
               ncol=3, frameon=False, fontsize=PLOT_FONT_PT, handletextpad=.4, columnspacing=.9)
    target = out / 'architecture_scatter.svg'
    save(fig, target)
    pd.concat(displayed, ignore_index=True).to_csv(out / 'architecture_scatter_points.csv', index=False)
    (out / 'architecture_scatter_statistics.json').write_text(json.dumps(cohorts, indent=2)+'\n')
    return target, cohorts


def text(parent, ident: str, x: float, y: float, lines: list[str], size=BODY_FONT_PT, step=18., bold=False, color='#111111'):
    """Place native editable poster text; sizes are final physical points."""
    el = ET.SubElement(parent, f'{{{SVG}}}text', {
        'id': ident, 'x': str(x), 'y': str(y),
        'style': f'font-family:DejaVu Sans;font-size:{size*PT_MM}px;font-weight:{"bold" if bold else "normal"};fill:{color}',
    })
    for i, line in enumerate(lines):
        ET.SubElement(el, f'{{{SVG}}}tspan', {'x': str(x), 'y': str(y+i*step)}).text = line
    return el


def wrap_lines(message: str, width_mm: float, point_size=BODY_FONT_PT, bold=False) -> list[str]:
    """Wrap measured glyph widths instead of estimating a character count."""
    properties = FontProperties(family='DejaVu Sans', size=point_size,
                                weight='bold' if bold else 'normal')
    measure = TextToPath()
    lines = []
    line = ''
    for word in message.split():
        trial = f'{line} {word}'.strip()
        width = measure.get_text_width_height_descent(trial, properties, False)[0] * PT_MM
        if width > width_mm and line:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def embed(parent, path: Path, ident: str, x: float, y: float, width: float) -> float:
    """Import vector SVG with local ID references renamed to avoid collisions."""
    source = ET.parse(path).getroot()
    _, _, w, h = map(float, source.attrib['viewBox'].split())
    mapping = {el.get('id'): ident+'-'+el.get('id') for el in source.iter() if el.get('id')}
    for el in source.iter():
        if el.get('id') in mapping:
            el.set('id', mapping[el.get('id')])
        for k, v in list(el.attrib.items()):
            v = re.sub(r'url\(#([^)]+)\)', lambda m: 'url(#'+mapping.get(m[1], m[1])+')', v)
            if v.startswith('#') and v[1:] in mapping:
                v = '#'+mapping[v[1:]]
            el.set(k, v)
    group = ET.SubElement(parent, f'{{{SVG}}}g', {
        'id': ident, f'{{{INK}}}label': path.stem,
        'transform': f'translate({x},{y}) scale({width/w})',
    })
    for child in source:
        if child.tag != f'{{{SVG}}}metadata':
            group.append(copy.deepcopy(child))
    return width*h/w


def compose(source: Path, out: Path, plots: list[Path], font: str, survey_cohorts: list[dict]) -> Path:
    source_digest = sha(source)
    # Namespace registration changes no retained element attributes or geometry.
    for _, ns in ET.iterparse(source, events=['start-ns']):
        prefix, uri = ns
        if prefix != 'svg':
            ET.register_namespace(prefix, uri)
    root = ET.parse(source).getroot()
    ids = {el.get('id') for el in root.iter() if el.get('id')}
    # A review can replace the generated layer in the current poster while
    # retaining any user edits elsewhere, rather than restoring an old input.
    if 'poster-right-20260929' in ids:
        removed = {'poster-right-20260929'}
    elif REMOVE <= ids:
        removed = REMOVE
    else:
        raise ValueError(f'Source does not match reviewed poster: missing {sorted(REMOVE-ids)}')
    for parent in list(root.iter()):
        for el in list(parent):
            if el.get('id') in removed:
                parent.remove(el)
    protected = {el.get('id'): hashlib.sha256(ET.tostring(el)).hexdigest()
                 for parent in root for el in parent if parent.tag == f'{{{SVG}}}g'}
    g = ET.SubElement(root, f'{{{SVG}}}g', {
        'id': 'poster-right-20260929', f'{{{INK}}}groupmode': 'layer',
        f'{{{INK}}}label': 'Right column · large type and DG-16 survey · 29 Sep',
    })
    x, width = 443., 383.
    text(g, 'right-exploration-heading', x, 220, ['1  Frontier design expands coverage'], HEADING_FONT_PT, bold=True)
    text(g, 'right-design-key', x, 245, ['C01 non-goal · C05 uniform goals · C15 frontier goals'], PLOT_FONT_PT)
    text(g, 'right-protocol', x, 262, ['100.04M frames · dots: 3 seeds; black segment: mean'], PLOT_FONT_PT)
    embed(g, plots[0], 'right-exploration-plot', x, 273, width)
    text(g, 'right-coverage-result', x, 369, ['C15 / C01: 2.1× coverage AUC; 2.7× cells.'], bold=True)
    text(g, 'right-exploration-caption', x, 391, [
        'A–B: last-10M means; AUC = mean cumulative cells.',
        'C–D: frozen probes; mobile windows, resets excluded.',
        '20-step mobility: 99.5% / 19.8% / 58.4% (C01/05/15).',
        'Mobile: path >500 units; return: displacement <100.',
    ])
    text(g, 'right-control-heading', x, 480, ['2  Broader coverage, weak goal control'], HEADING_FONT_PT, bold=True)
    embed(g, plots[1], 'right-control-plot', x, 508, width / 2)
    explanation = []
    for sentence in ['C15 lift <1 in all seeds.', 'Goal changes barely alter raw logits.',
                     'DG hits do not verify physical arrival.']:
        explanation.extend(wrap_lines(sentence, 180))
    text(g, 'right-control-explanation', x+203, 525, explanation)
    text(g, 'right-control-caption', x, 609, ['Shuffled baseline = 1.'])
    qualification = wrap_lines('Designs also differ in DG regularization and manager structure.', width)
    text(g, 'right-design-qualification', x, 642, qualification, color='#444444')
    text(g, 'right-survey-heading', x, 696, ['3  Which spatial code supports control?'], HEADING_FONT_PT, bold=True)
    text(g, 'right-survey-protocol', x, 721, [
        'DG 16 · 168 runs / 56 variants · saved ages: 25–150M frames',
    ], PLOT_FONT_PT)
    embed(g, plots[2], 'right-survey-plot', x, 733, width)
    text(g, 'right-survey-caption', x, 923, [
        'More peaks do not imply more target hits.',
        'Score–hit: CA3 feedback ρ=0.09; DG policy ρ≈0.',
        'Historical hit counters; recent spatial windows.',
    ])
    ET.SubElement(g, f'{{{SVG}}}rect', {
        'id': 'right-takehome-background', 'x': str(x), 'y': '988',
        'width': str(width), 'height': '133', 'rx': '2',
        'style': 'fill:#f2f2f2;stroke:#d0d0d0;stroke-width:0.6',
    })
    text(g, 'right-conclusion-heading', x+10, 1011, ['Take-home & discussion'], bold=True)
    text(g, 'right-conclusion', x+10, 1038, [
        'Broader exploration does not establish',
        'reliable landmark control.',
    ], bold=True)
    text(g, 'right-next-test', x+10, 1085, [
        'Which goal picks a place reliably:',
        'DG identity, or recent sequence context?',
    ])
    text(g, 'right-references-heading', x, 1140, ['References'], PLOT_FONT_PT, bold=True)
    text(g, 'right-references', x, 1155, [
        'Lin, Yiu & Leibold (2026), hippocampal sequence agent.',
        'Leibold (2020), Neural Networks.',
        'Raju et al. (2024), Science Advances; Wang et al. (2024), NeurIPS.',
    ], PLOT_FONT_PT, 14.)
    root.set(f'{{{SOD}}}docname', 'Bernstein2026_IntrMotiv_filled.svg')
    root.set(f'{{{INK}}}export-filename', 'Bernstein2026_IntrMotiv_filled.png')
    target = out / 'Bernstein2026_IntrMotiv_filled.svg'
    ET.ElementTree(root).write(target, encoding='utf-8', xml_declaration=True)
    written = ET.parse(target).getroot()
    lookup = {el.get('id'): el for el in written.iter() if el.get('id')}
    for ident, digest in protected.items():
        assert hashlib.sha256(ET.tostring(lookup[ident])).hexdigest() == digest, ident
    assert len([el.get('id') for el in written.iter() if el.get('id')]) == len(lookup), 'Duplicate SVG IDs'
    manifest = {
        'source_svg': str(source), 'source_svg_sha256': source_digest,
        'source_svg_hash_scope': 'Input before layer replacement; source and output may be the same path.',
        'output_svg': str(target.relative_to(ROOT)), 'font': font,
        'plot_source_script': str(Path(__file__).resolve().relative_to(ROOT)),
        'software': {'matplotlib': matplotlib.__version__, 'numpy': np.__version__, 'pandas': pd.__version__},
        'removed_right_objects': sorted(removed), 'retained_direct_subtree_count': len(protected),
        'retained_subtree_checks': 'pass', 'header_and_left_column': 'unchanged',
        'scalar_source': {'path': str(TABLE.relative_to(ROOT)), 'sha256': sha(TABLE)},
        'survey_source': {'path': str(SCATTER.relative_to(ROOT)), 'sha256': sha(SCATTER)},
        'survey_cohorts': survey_cohorts,
        'aggregation': 'Equal-weight seed means; all three seed observations displayed. No statistical-significance claims.',
        'survey_scope': 'DG 16 only; legacy 19x19 geometry; latest online windows with both spatial metrics and target-event counters. A/B show the same 168 runs (56 variants). Shape and color encode study family. DG 32/64 and other protocols screened separately in dg_capacity_audit.csv. No fitting or significance tests.',
        'typography': {'plot_labels_pt': PLOT_FONT_PT, 'explanatory_text_pt': BODY_FONT_PT,
                       'section_headings_pt': HEADING_FONT_PT},
        'selected_conditions': ['C01', 'C05', 'C15'], 'seeds': list(SEEDS), 'checkpoint_frames': FRAME,
        'revision': 'Large-type revision: 30 pt plot text, 40 pt explanation, smaller data areas, larger markers. Capacity fixed to DG 16; two online metric/control scatters replace the mixed-capacity four-panel survey. Visitor discussion asks how an event code becomes a physical goal.',
        'placeholders': [],
    }
    (out / 'build_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / '05_plans/poster_20260929')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(TABLE)
    data = data[data.condition.isin(COLORS)]
    assert len(data) == 9 and set(data.checkpoint_frames) == {FRAME}
    assert all(set(g.seed) == set(SEEDS) for _, g in data.groupby('condition'))
    font = style()
    plots, sources = render_plots(data, args.output)
    print(compose(args.source.resolve(), args.output.resolve(), plots, font, sources))


if __name__ == '__main__':
    main()
