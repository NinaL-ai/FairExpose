import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

files = [
    "results/ranking/colorblind_non-binary_k100_t0_top_k_df.csv",
    "results/ranking/FairPPCO_non-binary_k100_t0_top_k_df.csv",
    "results/ranking/FairOPopt_non-binary_k100_t0_top_k_df.csv",
]

# --- paper-style fonts ---
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})

# --- colorblind-safe palette (4 groups max) ---
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)

all_lines = []
all_labels = []

for i, file in enumerate(files):
    df = pd.read_csv(file)
    df = df.reset_index(drop=True)
    df["position"] = df.index + 1
    df["exposure"] = 1 / np.log2(df["position"] + 1)

    # stable group order
    groups = sorted(df["z"].unique(), key=lambda x: int(x))

    ax = axes[i]

    for j, g in enumerate(groups):
        mask = df["z"] == g
        group_exposure = df["exposure"].where(mask, 0.0)
        cum = group_exposure.cumsum()

        line, = ax.plot(
            df["position"],
            cum,
            label=str(g),
            color=colors[j % len(colors)],
            linewidth=2
        )

        if i == 0:
            all_lines.append(line)
            all_labels.append(str(g))

    method = file.split("/")[-1].split("_")[0]
    if method == "colorblind":
        method = "Colorblind"
    elif method == "FairPPCO":
        method = "FairExpPro"
    elif method == "FairOPopt":
        method = "FairExpOrd-Exact"

    ax.set_title(method)
    ax.set_xlabel(r"Position $[p]$")

    if i == 0:
        ax.set_ylabel(r"Exposure [$E(\tau_p,g)$]")

    ax.tick_params(axis='y', labelleft=True)

    # paper-style grid
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.60)
    ax.set_xlim(1, df["position"].max())
    ax.set_xticks([1, 20, 40, 60, 80, 100])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)

    if i > 0:
        ax.tick_params(axis='y', labelleft=False)

plt.tight_layout()

# --- leave space for right-side legend ---
plt.subplots_adjust(right=0.82)

# --- global legend (RIGHT SIDE) ---
fig.legend(
    all_lines,
    all_labels,
    title="Group",
    loc="center left",
    bbox_to_anchor=(0.83, 0.55),
    frameon=True,
    fancybox=False,
    edgecolor="0.6"
)

plt.savefig("experiments/exp1_optimality/results/exp1.pdf")
plt.show()