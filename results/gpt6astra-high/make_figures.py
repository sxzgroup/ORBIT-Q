#!/usr/bin/env python3
"""Two figures comparing Astra with the eight frozen campaigns in PR #28."""
import json
from io import StringIO
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "../../.tmp/mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

data = json.loads((OUT / "comparison.json").read_text())
models = data["models"]
assert models["astra_high"]["completed_tasks"] == 12
destination = OUT / "figs"
destination.mkdir(exist_ok=True)

public = json.loads((OUT / "public-model-comparison.json").read_text())
RUNS = public["runs"] + [{
    "label": "Astra high",
    "passes": [row["task"] for row in models["astra_high"]["tasks"] if row["raw_reward"] == 1],
    "agent_wall_sec": models["astra_high"]["agent_wall_sec"],
    "total_cost_usd": models["astra_high"]["cost_usd"],
}]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#B2182B",
          "#111111", "#56B4E9", "#8C564B", "#7B2CBF"]
assert len(RUNS) == len(COLORS) == 9
for key, run in zip(["sol_high", "sol_ultra"], RUNS[:2]):
    assert run["passes"] == [row["task"] for row in models[key]["tasks"] if row["final_valid"]]
    assert np.isclose(run["agent_wall_sec"], models[key]["agent_wall_sec"])
    assert np.isclose(run["total_cost_usd"], models[key]["cost_usd"])
TASKS = np.arange(1, 13)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 240, "figure.facecolor": "white",
    "svg.hashsalt": "orbitq-astra-high", "svg.fonttype": "none", "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(destination / f"{name}.png", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(destination / f"{name}.pdf", bbox_inches="tight", pad_inches=0.08,
                metadata={"CreationDate": None, "ModDate": None})
    svg = StringIO()
    fig.savefig(svg, format="svg", bbox_inches="tight", pad_inches=0.08, metadata={"Date": None})
    # Matplotlib adds trailing spaces to SVG path lines. Keep tracked exports clean.
    (destination / f"{name}.svg").write_text("\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n")
    plt.close(fig)


def outcome_figure():
    fig, ax = plt.subplots(figsize=(10.8, 5.7))
    matrix = np.array([[task in run["passes"] for task in TASKS] for run in RUNS])
    ax.imshow(matrix, aspect="auto", cmap=ListedColormap(["#FFFFFF", "#D9ECE7"]),
              vmin=0, vmax=1)
    for row in range(len(RUNS)):
        for col in range(12):
            ax.text(col, row, "P" if matrix[row, col] else "F",
                    color="#175B48" if matrix[row, col] else "#333333",
                    ha="center", va="center", weight="bold", fontsize=10)
    labels = [
        f"{run['label']}{'*' if run['label'] == 'Astra high' else ''} · {len(run['passes'])}/12"
        for run in RUNS
    ]
    ax.set_xticks(np.arange(12), [f"{task:02d}" for task in TASKS])
    ax.set_yticks(np.arange(len(RUNS)), labels)
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(RUNS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    ax.set_xlabel("Challenge")
    ax.set_title("TensorCircuit solver comparison: task outcomes",
                 fontsize=14, weight="bold", pad=16)
    fig.text(0.01, 0.02,
             "P = passing solution; F = failed task. Historical rows: published final outcomes. *Astra: automatic only, no human adjudication.",
             fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "astra-high-vs-sol-outcomes")


def resource_figure():
    astra = models["astra_high"]
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.6),
                             gridspec_kw={"width_ratios": [1, 1, 1.35]})
    wall = np.array([row["agent_wall_sec"] / 60 for row in astra["tasks"]])
    axes[0].bar(TASKS, wall, color="#009E73", edgecolor="black", linewidth=0.6, width=0.72)
    axes[0].set(ylabel="Agent solve time (min)", ylim=(0, max(wall) * 1.2))
    axes[0].set_title(f"(a) Solver wall time · {sum(wall):.2f} min", loc="left", fontsize=11)
    bottom = np.zeros(12)
    for field, label, color in [
        ("non_cache_input_tokens", "Non-cache-read prompt", "#595959"),
        ("cache_tokens", "Cache-read prompt", "#56B4E9"),
        ("output_tokens", "Output", "#D55E00"),
    ]:
        values = np.array([row[field] for row in astra["tasks"]]) / 1e6
        axes[1].bar(TASKS, values, bottom=bottom, color=color, label=label,
                    width=0.72, linewidth=0.4, edgecolor="black")
        bottom += values
    assert np.isclose(bottom.sum(), astra["total_tokens"] / 1e6)
    axes[1].set(ylabel="Solving-side tokens (million)", ylim=(0, max(bottom) * 1.4))
    axes[1].set_title(f"(b) Solver token use · {bottom.sum():.3f}M", loc="left", fontsize=11)
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    for ax in axes[:2]:
        ax.set_xticks(TASKS, [f"{n:02d}" for n in TASKS])
        ax.set_xlabel("Challenge")
    label_positions = [(28, 2.68), (27, 3.15), (30, 1.47), (29, 0.40),
                       (45, 0.65), (25, 0.96), (63, 0.88), (30, 2.12), (4, 1.65)]
    for run, color, position in zip(RUNS, COLORS, label_positions):
        count = len(run["passes"])
        x = run["agent_wall_sec"] / 60 / count
        y = run["total_cost_usd"] / count
        axes[2].scatter(x, y, s=run["total_cost_usd"] * 28, color=color,
                        edgecolor="black", linewidth=0.9, zorder=3)
        label = run["label"].replace("DeepSeek V4 ", "DeepSeek V4\n")
        axes[2].annotate(f"{label}\n({count}/12)", (x, y), xytext=position,
                         textcoords="data", ha="left", va="center",
                         fontsize=8, color=color, weight="bold")
    axes[2].set(xlabel="Solve time per passing solution (min)",
                ylabel="Solver cost per passing solution (USD)",
                xlim=(0, 80), ylim=(-0.12, 3.85))
    axes[2].set_title("(c) Configuration-level resource efficiency", loc="left", fontsize=11)
    axes[2].text(0.98, 0.98, "Marker area ∝ total solver cost",
                  transform=axes[2].transAxes, ha="right", va="top", fontsize=7)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle("GPT-6 Astra high agent-side resource use (TensorCircuit 1.8)",
                  x=0.01, ha="left", fontweight="bold", fontsize=14)
    fig.text(0.01, 0.02,
             "Per-pass basis: historical final outcomes; Astra automatic only. All attempts included; verifier cost/time excluded. Prices are not normalized.",
             fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    save(fig, "astra-high-agent-resource-use")


if __name__ == "__main__":
    outcome_figure()
    resource_figure()
    print("Generated two figures with all eight PR #28 campaigns plus Astra (purple); panels a/b unchanged.")
