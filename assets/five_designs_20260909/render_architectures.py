"""Render conceptual architecture diagrams; no experimental data or simulations."""
from pathlib import Path
import subprocess
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', tempfile.mkdtemp(prefix='intrmotiv-figures-'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ft2font import FT2Font
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
FONT = Path(subprocess.check_output(
    ['fc-match', '-f', '%{file}', 'DejaVu Sans'], text=True).strip())
if not FONT.is_file() or FONT.suffix.lower() not in {'.ttf', '.otf'}:
    raise RuntimeError('A verified scalable font is required')
FT2Font(str(FONT))
FP = FontProperties(fname=str(FONT), size=16)
plt.rcParams['svg.fonttype'] = 'path'

PLANS = [
    ('Anchored goal control', ['Frozen goals', 'Controlled graph', 'Goal worker'],
     ['sensory snippets', '+ frontier queue', 'cost → action'],
     'Control error trains DG; graph search composes local skills.'),
    ('Predictive contextual states', ['Categorical state', 'Outcome planner', 'Goal worker'],
     ['+ test predictor', '+ frontier queue', 'cost → action'],
     'DG preserves distinctions needed to predict controlled outcomes.'),
    ('Replay of routes to many goals', ['Anchored replay', 'All-goal critic', 'Action selection'],
     ['real transitions', 'Bellman updates', 'lowest goal cost'],
     'Replay trains route costs; no graph search at decision time.'),
    ('Distinguishable intrinsic skills', ['Endpoint decoder', 'Validated skills', 'Skill worker'],
     ['decode command', '+ frontier queue', 'skill → action'],
     'Discover distinct outcomes, then refine reliable arrival speed.'),
    ('Local reachability landmarks', ['Anchors + trials', 'Sparse local cover', 'Goal worker'],
     ['both directions', '+ frontier queue', 'cost → action'],
     'DG detects controllable neighborhoods; locality is explicit.'),
]

def draw(index, title, names, descriptions, footer):
    # Avoid one-pixel truncation from floating-point inch-to-pixel conversion.
    fig = plt.figure(figsize=(8.2+1e-9, 4.35+1e-9), dpi=100, facecolor='#ffffff')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 820), ylim=(435, 0))
    ax.axis('off')
    labels = []
    ax.text(24, 32, f'{index:02d}  {title}', fontproperties=FP,
            fontsize=19, color='#142d40', va='center')
    centers = [144, 410, 676]
    def box(cx, y, title_, detail, fill):
        patch = FancyBboxPatch((cx-120, y), 240, 83,
            boxstyle='round,pad=0,rounding_size=12', lw=1.25,
            edgecolor='#a5b8c5', facecolor=fill)
        ax.add_patch(patch)
        for yy, label in [(y+29, title_), (y+59, detail)]:
            text = ax.text(cx, yy, label, ha='center', va='center',
                fontproperties=FP, color='#142d40')
            labels.append((text, patch))
    def arrow(start, end, bend=0):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
            mutation_scale=16, lw=1.8, color='#43677f',
            connectionstyle=f'arc3,rad={bend}'))
    box(144, 74, 'Observation', 'fixed visual trunk', '#eef3f6')
    box(410, 74, 'Sparse DG', 'contextual' if index==5 else 'learned detector', '#dcf0e8')
    box(676, 74, 'Fixed CA3', 'sequence memory', '#e4eaf9')
    arrow((268, 116), (286, 116))
    arrow((534, 116), (552, 116))
    for cx, name, desc in zip(centers, names, descriptions):
        box(cx, 243, name, desc, '#fcf0dc')
    arrow((268, 285), (286, 285))
    arrow((534, 285), (552, 285))
    arrow((676, 160), (676, 239))
    if index==2:
        # Additional readout from CA3 into categorical predictive state.
        ax.plot([641, 641, 144, 144], [159, 200, 200, 223],
                color='#43677f', lw=1.8)
        arrow((144, 222), (144, 239))
    elif index==3:
        ax.plot([641, 641, 410, 410], [159, 200, 200, 223],
                color='#43677f', lw=1.8)
        arrow((410, 222), (410, 239))
    elif index==5:
        # Prior CA3 context enters the detector, never an instantaneous cycle.
        ax.plot([600, 600, 410], [160, 196, 196], color='#43677f', lw=1.8)
        arrow((410, 196), (410, 160))
        ax.text(513, 220, 'prior context', ha='center', fontproperties=FP,
                color='#43677f', fontsize=16)
    ax.text(24, 367, 'LEARNING PRINCIPLE', fontproperties=FP,
            color='#496176', fontsize=16)
    # Deliberately wrap at semantic boundaries for report-width reading.
    breaks = {
        1: 'Control error trains DG; graph search\ncomposes local skills.',
        2: 'DG preserves distinctions needed to predict\ncontrolled outcomes.',
        3: 'Replay trains route costs; no graph search\nat decision time.',
        4: 'Discover distinct outcomes, then refine\nreliable arrival speed.',
        5: 'DG detects controllable neighborhoods;\nlocality is explicit.',
    }
    ax.text(24, 394, breaks[index], fontproperties=FP,
            color='#142d40', va='center', linespacing=1.3)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label, patch in labels:
        lb, pb = label.get_window_extent(renderer), patch.get_window_extent(renderer)
        if lb.x0 < pb.x0+3 or lb.x1 > pb.x1-3 or lb.y0 < pb.y0 or lb.y1 > pb.y1:
            raise RuntimeError(f'Label outside box: {label.get_text()}')
    fig.savefig(OUT / f'plan_{index}.png', dpi=200)
    fig.savefig(OUT / f'plan_{index}.svg')
    # This is the expected Markdown display size, used for visual inspection.
    fig.savefig(OUT / f'plan_{index}_preview.png', dpi=100)
    plt.close(fig)

if __name__ == '__main__':
    for i, args in enumerate(PLANS, 1):
        draw(i, *args)
    print(f'Rendered five diagrams with scalable font: {FONT}')
