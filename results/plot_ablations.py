"""Generate the ARR ablation overview directly from the consolidated parquet."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "arr_ablations_all.parquet"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

MODELS = ["llama3_8b", "mistral_7b", "qwen2_5_7b"]
DATASETS = ["chaosnli", "chaosnli_alpha", "pavlick_nli"]
MODEL_SHORT = {"llama3_8b": "Llama", "mistral_7b": "Mistral", "qwen2_5_7b": "Qwen"}
DATASET_SHORT = {"chaosnli": "Chaos", "chaosnli_alpha": "Alpha", "pavlick_nli": "Pavlick"}
COLORS = {"raw": "#4C72B0", "controlled": "#DD8452", "mean": "#4C72B0",
          "first": "#55A868", "null": "#8C8C8C"}


def ordered(df):
    df = df.copy()
    df["order"] = (df.model.map({m: i for i, m in enumerate(MODELS)}) * 3
                   + df.dataset.map({d: i for i, d in enumerate(DATASETS)}))
    return df.sort_values("order")


def labels(df):
    return [f"{MODEL_SHORT[m]} / {DATASET_SHORT[d]}" for m, d in zip(df.model, df.dataset)]


df = pd.read_parquet(DATA)
fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.2))
fig.subplots_adjust(hspace=0.48, wspace=0.32)

# A: raw and length-controlled correlations with uncertainty.
ax = axes[0, 0]
d = ordered(df[(df.ablation == "length") & (df.signal == "credal_width")])
y = np.arange(len(d)); off = 0.13
ax.errorbar(d.rho_H, y - off, fmt="o", color=COLORS["raw"], label="Raw rho(W,H)", zorder=3)
ax.errorbar(d.partial, y + off,
            xerr=np.vstack([d.partial - d.ci_lo, d.ci_hi - d.partial]),
            fmt="s", color=COLORS["controlled"], capsize=2.5,
            label="Partial rho(W,H | length)", zorder=3)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(y, labels(d)); ax.invert_yaxis(); ax.set_xlim(-0.56, 0.13)
ax.set_xlabel("Spearman correlation")
ax.legend(frameon=False, fontsize=8, loc="lower right")

# B: pooling intervention.
ax = axes[0, 1]
d = ordered(df[(df.ablation == "pool") & (df.signal == "credal_width")])
p = d.pivot(index=["model", "dataset"], columns="scheme", values="rho_H").reset_index()
p = ordered(p); y = np.arange(len(p))
for yi, a, b in zip(y, p.span_mean, p.first_token):
    ax.plot([a, b], [yi, yi], color="#B8B8B8", linewidth=1.2, zorder=1)
ax.scatter(p.span_mean, y, color=COLORS["mean"], marker="o", label="span_mean", zorder=3)
ax.scatter(p.first_token, y, color=COLORS["first"], marker="D", label="first_token", zorder=3)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(y, labels(p)); ax.invert_yaxis(); ax.set_xlim(-0.56, 0.13)
ax.set_xlabel("rho(W,H)")
ax.legend(frameon=False, fontsize=8, loc="lower right")

# C: observed effect against the permutation null's 95th percentile.
ax = axes[1, 0]
d = ordered(df[(df.ablation == "shuffle") & (df.signal == "credal_width")])
y = np.arange(len(d))
ax.barh(y, d.null_p95, color="#D0D0D0", height=0.62, label="95th percentile |null rho|")
ax.scatter(np.abs(d.rho_obs), y, color=COLORS["controlled"], marker="o",
           label="Observed |rho(W,H)|", zorder=3)
ax.set_yticks(y, labels(d)); ax.invert_yaxis(); ax.set_xlim(0, 0.45)
ax.set_xlabel("Absolute Spearman correlation")
ax.legend(frameon=False, fontsize=8, loc="lower right")

# D: design-grid range, retaining sign information.
ax = axes[1, 1]
d = df[(df.ablation == "grid") & (df.signal == "credal_width")]
g = (d.groupby(["model", "dataset"], as_index=False).rho
     .agg(rho_min="min", rho_max="max"))
g = ordered(g); y = np.arange(len(g))
for yi, lo, hi, dataset in zip(y, g.rho_min, g.rho_max, g.dataset):
    color = "#4C72B0" if dataset == "chaosnli" else "#8C8C8C"
    ax.plot([lo, hi], [yi, yi], color=color, linewidth=3, solid_capstyle="round")
    ax.scatter([lo, hi], [yi, yi], color=color, s=22, zorder=3)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(y, labels(g)); ax.invert_yaxis(); ax.set_xlim(-0.66, 0.34)
ax.set_xlabel("Range of rho(W,H) over K and layer")

for ax in axes.flat:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#E7E7E7", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    ax.tick_params(labelsize=8.5)

fig.savefig(OUT / "ablations_overview.png", dpi=220, bbox_inches="tight")
fig.savefig(OUT / "ablations_overview.pdf", bbox_inches="tight")
print(OUT / "ablations_overview.png")
