"""Reproduce selected outlier figures from cached W&B and canonical snapshots.

Run with the desktop SF_git Python. Lines are window means, not independent
replicates; no inferential error bars are assigned to repeated training logs.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results/late_outliers_20260908"
FONT = Path(font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"), fallback_to_default=False))
assert FONT.is_file() and FONT.suffix.lower() in {".ttf", ".otf"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 18,
    "axes.labelsize": 18, "axes.titlesize": 19, "xtick.labelsize": 16,
    "ytick.labelsize": 16, "legend.fontsize": 16, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False})

def save(fig, name):
    fig.savefig(DATA / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(DATA / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)

snap = pd.read_csv(DATA / "dgp_snapshots/per_snapshot.csv")
focal = "DGP_C15_HIT_JOINT_LEG_S123"
fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
for seed, color, style in [(123, "#0072B2", "-"), (8, "#D55E00", "--"), (99, "#777777", ":")]:
    name = f"DGP_C15_HIT_JOINT_LEG_S{seed}"
    z = snap[snap.run_name == name].sort_values("target_env_steps")
    axes[0,0].plot(z.target_env_steps/1e6, z.mono_field_unit_fraction*100,
        style, color=color, marker="o", label=f"Seed {seed}")
    axes[0,1].plot(z.target_env_steps/1e6, z.active_only_map_cosine,
        style, color=color, marker="o")
axes[0,0].set(title="a  Single-field units", ylabel="Eligible units (%)", ylim=(-2, 35))
axes[0,0].legend(frameon=False)
axes[0,1].set(title="b  Overlap between DG maps", ylabel="Active-only map cosine", ylim=(0, .45))
b = pd.read_csv(DATA / f"{focal}_verified_behavior.csv.gz")
metric = "policy_stats/avg_z_00_openfield_map2_fixed_loc3_fixedlength_noreward_coverage_auc"
rows=[]
for low in range(0, 75, 5):
    w=b[b['train/env_steps'].between(low*1e6,(low+5)*1e6,inclusive="left")]
    def rate(prefix):
        return w[f"intrmotiv/hrl/{prefix}_hit_numerator"].sum()/w[f"intrmotiv/hrl/{prefix}_hit_event_count"].sum()
    rows.append({"mid":low+2.5, "coverage":w[metric].mean(), "lift":rate("target")/rate("shuffled")})
v=pd.DataFrame(rows)
axes[1,0].plot(v.mid,v.coverage,"o-",color="#0072B2")
axes[1,0].set(title="c  Focal run: exploration", ylabel="Episode coverage AUC", ylim=(0,100))
axes[1,1].plot(v.mid,v.lift,"o-",color="#0072B2")
axes[1,1].axhline(1,color="#555555",ls="--",lw=1.2)
axes[1,1].set(title="d  Focal run: goal specificity", ylabel="Target / shuffled hit rate", ylim=(.94,1.06))
for ax in axes.flat:
    ax.set(xlabel="Training steps (millions)", xlim=(0,77), xticks=[0,25,50,75])
save(fig,"focal_learning")

# Every point is one matched seed/outcome, so the interface interaction is visible.
final=snap[snap.target_env_steps==75000000]
fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
pair_rows=[]
for x,(goal,label,color) in enumerate([("legacy","Legacy","#0072B2"),("target_id_film","FiLM","#D55E00")]):
    g=final[final.goal_conditioning==goal]
    a=g[g.ppo_dg_gradient=="joint"].set_index(["worker_outcome","seed"])
    b=g[g.ppo_dg_gradient=="stop"].set_index(["worker_outcome","seed"])
    for ax,metric,scale in [(axes[0],"mono_field_unit_fraction",100),(axes[1],"active_only_map_cosine",1)]:
        d=(a[metric]-b[metric]).sort_index()*scale
        ax.scatter(x+np.linspace(-.13,.13,len(d)),d,color=color,s=55)
        ax.plot([x-.2,x+.2],[d.mean()]*2,color=color,lw=3)
        for (outcome,seed),value in d.items():
            pair_rows.append(dict(goal=goal,outcome=outcome,seed=seed,metric=metric,delta=value,scale=scale))
for ax in axes:
    ax.axhline(0,color="#777777",ls="--",lw=1)
    ax.set(xticks=[0,1],xticklabels=["Legacy","FiLM"],xlim=(-.5,1.5))
axes[0].set(title="a  Single-field fraction",ylabel="JOINT − STOP (percentage points)")
axes[1].set(title="b  Map overlap",ylabel="JOINT − STOP (cosine)")
save(fig,"gradient_interface_pairs")
pd.DataFrame(pair_rows).to_csv(DATA/"snapshot_paired_effects.csv",index=False)

# Three explicitly selected examples: two final mono-field units and one
# non-mono control. Same raw occupancy-normalized map scale across checkpoints.
maps=json.loads((DATA/"focal_map_summaries.json").read_text())
by_target={r["target"]:r for r in maps}
end=by_target[75000000]
mono=np.flatnonzero(end["field_mono"])
nonmono=np.flatnonzero(~np.asarray(end["field_mono"],dtype=bool))
units=[int(mono[0]),int(mono[1]),int(nonmono[0])]
fig,axes=plt.subplots(3,2,figsize=(10,12),constrained_layout=True)
cmap=plt.get_cmap("viridis").copy();cmap.set_bad("#dddddd")
for row,unit in enumerate(units):
    vmax=max(np.asarray(by_target[t]["rate_maps"])[:,:,unit].max() for t in [25000000,75000000])
    for col,target in enumerate([25000000,75000000]):
        d=by_target[target];m=np.asarray(d["rate_maps"])[:,:,unit];occ=np.asarray(d["occupancy"])
        ax=axes[row,col];im=ax.imshow(np.ma.array(m,mask=occ==0),origin="lower",extent=d["bounds"],cmap=cmap,vmin=0,vmax=vmax,interpolation="nearest")
        kind="single-field" if d["field_mono"][unit] else "multi-field"
        ax.set(title=f"Unit {unit} · {target//1000000}M\n{kind}",xlabel="x (environment units)",ylabel="y (environment units)")
        ax.set_xticks([d["bounds"][0],d["bounds"][1]]);ax.set_yticks([d["bounds"][2],d["bounds"][3]])
    fig.colorbar(im,ax=axes[row,:],shrink=.75,label="Mean DG activity")
save(fig,"focal_map_examples")
(DATA/"figure_manifest.json").write_text(json.dumps({"font":str(FONT),"matplotlib":matplotlib.__version__,"numpy":np.__version__,"map_example_units":units,"map_selection":"first two final mono-field indices plus first final non-mono index; not representative of all 16 units","spatial_source":"canonical 100k-sample milestone snapshots","behavior_source":"unsampled W&B scan_history; 5M means and ratios of summed counts","uncertainty":"none; exploratory selection, dependent training observations"},indent=2))
