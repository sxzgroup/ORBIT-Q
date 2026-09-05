#!/usr/bin/env python3
"""Generate vector-ready ORBIT-Q paper-update figures.

The layouts deliberately follow arXiv:2607.03105: Fig. 1c is an
agent-by-framework matrix, Fig. 2b is the TC agent-axis failure/runtime
scatter, and Fig. 4 keeps the original 2 x 3 resource-use structure.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data"
OUT = ROOT / "updated_figures"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "grid.color": "#D6D6D6",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.70,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)

COLORS = {
    "gpt55": "#0072B2",
    "gpt55_checklist": "#56B4E9",
    "opus48": "#D55E00",
    "glm52": "#A05195",
    "sonnet46": "#E69F00",
    "sol56_high": "#2563EB",
    "sol56_ultra": "#F97316",
    "terra56_high": "#009E73",
    "luna56_high": "#CC79A7",
    "deepseek_v4_flash_high": "#4D4D4D",
    "grok45_high": "#111111",
    "deepseek_v4_pro_high": "#007F87",
    "grok46_high": "#566573",
    "astra_high": "#7B2CBF",
    "tensorcircuit": "#0072B2",
    "pennylane": "#CC79A7",
    "torchquantum": "#E69F00",
    "mindquantum": "#009E73",
}

PASS = "#009E73"
FAIL = "#FFFFFF"
CACHE = "#56B4E9"


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in (None, "") else np.nan


def clean_axes(ax: mpl.axes.Axes, *, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.11,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save_all(fig: mpl.figure.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def validate_new_slowdowns(agent_rows: list[dict[str, str]]) -> None:
    """Check archived fixed-paper ratios; these no longer drive Fig. 2b."""
    refs = {int(r["task"]): num(r, "expert_tc_runtime_sec") for r in read_csv("paper_expert_runtimes.csv")}
    outcomes = read_csv("task_outcomes.csv")
    model_to_agent = {
        "gpt56_sol_high": "sol56_high",
        "gpt56_sol_ultra": "sol56_ultra",
        "gpt56_terra_high": "terra56_high",
        "gpt56_luna_high": "luna56_high",
        "deepseek_v4_flash_high": "deepseek_v4_flash_high",
        "grok45_high": "grok45_high",
        "deepseek_v4_pro_high": "deepseek_v4_pro_high",
        "grok46_high": "grok46_high",
        "astra_high": "astra_high",
    }
    expected = {r["key"]: num(r, "gm_slowdown") for r in agent_rows if r["series"] == "new"}
    for model, agent_key in model_to_agent.items():
        ratios = [
            num(r, "runtime_sec") / refs[int(r["task"])]
            for r in outcomes
            if r["model"] == model and r["final_pass"] == "1"
        ]
        recomputed = math.exp(sum(math.log(v) for v in ratios) / len(ratios))
        if not math.isclose(recomputed, expected[agent_key], rel_tol=0.0, abs_tol=5e-7):
            raise ValueError(f"{model}: stored GM {expected[agent_key]} != recomputed {recomputed}")


def make_fig1c(agent_rows: list[dict[str, str]], framework_rows: list[dict[str, str]]) -> None:
    agents = [r for r in agent_rows if r["include_fig1c"] == "yes"]
    order = ["gpt55", "opus48", "glm52", "sonnet46", "sol56_high", "sol56_ultra", "terra56_high", "luna56_high", "deepseek_v4_flash_high", "grok45_high", "deepseek_v4_pro_high", "grok46_high", "astra_high"]
    agents = sorted(agents, key=lambda r: order.index(r["key"]))
    frameworks = {r["key"]: r for r in framework_rows}

    values = np.full((4, len(agents)), np.nan)
    values[0, :] = [int(r["passes"]) for r in agents]
    values[1, 0] = int(frameworks["pennylane"]["passes"])
    values[2, 0] = int(frameworks["torchquantum"]["passes"])
    values[3, 0] = int(frameworks["mindquantum"]["passes"])

    cmap = LinearSegmentedColormap.from_list(
        "orbit_passes", ["#E9B5AB", "#F3D7A2", "#CFE1C9", "#B8D8D8"]
    )
    norm = Normalize(3, 12)
    fig, ax = plt.subplots(figsize=(10.2, 2.75))
    ax.set_xlim(-0.85, len(agents))
    ax.set_ylim(4.25, -1.1)
    ax.axis("off")

    for i in range(4):
        for j in range(len(agents)):
            value = values[i, j]
            color = "#F7F7F7" if np.isnan(value) else cmap(norm(value))
            edge = "#D8D8D8" if np.isnan(value) else mpl.colors.to_hex(np.array(mpl.colors.to_rgb(color)) * 0.82)
            ax.add_patch(Rectangle((j + 0.06, i + 0.06), 0.88, 0.88, facecolor=color, edgecolor=edge, linewidth=1.0))
            text = "–" if np.isnan(value) else f"{int(value)}/12"
            ax.text(j + 0.50, i + 0.50, text, ha="center", va="center", fontweight="bold" if not np.isnan(value) else "normal", color="#333333")

    row_labels = ["TC", "PL", "TQ", "MQ"]
    for i, label in enumerate(row_labels):
        ax.text(-0.08, i + 0.50, label, ha="right", va="center", fontsize=9.5)

    labels = [
        "GPT-5.5",
        "Opus-4.8",
        "GLM-5.2",
        "Sonnet-4.6",
        "5.6 Sol",
        "5.6 Sol\nultra",
        "5.6 Terra",
        "5.6 Luna",
        "DeepSeek\nV4 Flash",
        "Grok 4.5",
        "DeepSeek\nV4 Pro",
        "Grok 4.6",
        "GPT-6\nAstra",
    ]
    for j, label in enumerate(labels):
        ax.text(j + 0.50, -0.02, label, ha="center", va="bottom", fontsize=7.7,
                color=COLORS["astra_high"] if agents[j]["key"] == "astra_high" else "#222222")

    ax.text(-0.68, 2.0, "Framework", rotation=90, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#666666")
    ax.set_title("Updated agent–framework benchmarking matrix", loc="left", fontweight="bold", pad=18)
    save_all(fig, "fig1c_updated_agent_framework_matrix")


def make_fig2b(agent_rows: list[dict[str, str]]) -> None:
    rows = [r for r in agent_rows if r["include_fig2b"] == "yes"]
    paired = read_csv("astra_expert_runtime.csv")
    paired_ratios = [num(r, "astra_sec") / num(r, "expert_reference_sec")
                     for r in paired]
    astra_ratio = math.exp(sum(math.log(v) for v in paired_ratios) / len(paired_ratios))
    fig, ax = plt.subplots(figsize=(5.3, 4.4))
    clean_axes(ax)
    ax.axhline(1.0, color="#888888", ls=(0, (3, 3)), lw=0.9, zorder=1)
    ax.axvline(0.0, color="#888888", ls=(0, (3, 3)), lw=0.9, zorder=1)
    # The diamond is the normalization anchor, not a measured failure rate
    # for the expert (Astra's Task 08 expert run did not complete).
    ax.scatter([0], [1], marker="D", s=82, facecolor="none", edgecolor="#888888", zorder=2)
    ax.text(1.3, 1.40, "Expert TC\nreference", fontsize=7.5, color="#777777", ha="left", va="center")

    # Preserve the original paper's four agent colors and label positions.
    # Additional configurations use distinct colors in the same axis.
    fig2_colors = {
        "gpt55": "#0072B2",
        "opus48": "#E69F00",
        "sonnet46": "#CC79A7",
        "glm52": "#D55E00",
        "sol56_high": "#A65628",
        "sol56_ultra": "#009E73",
        "terra56_high": "#56B4E9",
        "luna56_high": "#8C6D31",
        "deepseek_v4_flash_high": "#4D4D4D",
        "grok45_high": "#111111",
        "deepseek_v4_pro_high": COLORS["deepseek_v4_pro_high"],
        "grok46_high": COLORS["grok46_high"],
        "astra_high": COLORS["astra_high"],
    }
    label_positions = {
        "gpt55": (18.2, 1.30, "left"),
        "opus48": (24.0, 3.70, "right"),
        "sonnet46": (47.2, 8.15, "left"),
        "glm52": (57.0, 7.10, "left"),
        "sol56_high": (14.6, 3.00, "right"),
        "sol56_ultra": (14.2, 2.15, "right"),
        "terra56_high": (29.5, 2.78, "left"),
        "luna56_high": (28.2, 5.00, "left"),
        "deepseek_v4_flash_high": (58.3, 4.08, "center"),
        "grok45_high": (36.0, 4.12, "left"),
        "deepseek_v4_pro_high": (58.3, 2.02, "center"),
        "grok46_high": (30.0, 2.00, "left"),
        "astra_high": (3.5, 0.08, "left"),
    }
    for row in rows:
        x = 100.0 * int(row["failures"]) / 12.0
        y = astra_ratio if row["key"] == "astra_high" else num(row, "gm_slowdown")
        color = fig2_colors[row["key"]]
        ax.scatter(x, y, marker="o", s=34, facecolor=color, edgecolor="#222222", linewidth=0.8, alpha=0.9, zorder=3)
        label_x, label_y, align = label_positions[row["key"]]
        label = f"{row['short_label']}\n({row['passes']}/12)"
        ax.text(
            label_x,
            label_y,
            label,
            fontsize=7.2,
            fontweight="bold",
            color=color,
            ha=align,
            va="center",
        )

    # Negative limits provide visual padding only; data coordinates are unchanged.
    ax.set_xlim(-10, 72)
    ax.set_ylim(-0.6, 9.2)
    ax.set_xticks(np.arange(0, 71, 10))
    ax.set_yticks([0, 1, 3, 5, 7, 9])
    ax.set_xlabel("Failure rate (%)")
    ax.set_ylabel("Runtime / expert TC reference")
    panel_label(ax, "(b)")
    save_all(fig, "fig2b_updated_agent_axis")


def make_fig4(agent_rows: list[dict[str, str]], framework_rows: list[dict[str, str]]) -> None:
    fig = plt.figure(figsize=(16.0, 9.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.15],
                          height_ratios=[1.25, 1], wspace=0.50, hspace=0.42)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    axa, axb, axc, axd, axe, axf = axes

    # Agent axis: resource totals for every reported configuration.
    y = np.arange(len(agent_rows))
    passed_h = np.array([num(r, "wall_passed_sec") / 3600 for r in agent_rows])
    failed_h = np.array([num(r, "wall_failed_sec") / 3600 for r in agent_rows])
    axa.barh(y, passed_h, color=PASS, edgecolor="#222222", linewidth=0.7, label="Passed tasks", zorder=2)
    axa.barh(y, failed_h, left=passed_h, color=FAIL, edgecolor="#222222", linewidth=0.7, label="Failed tasks", zorder=2)
    axa.set_yticks(y, [r["short_label"] for r in agent_rows])
    axa.invert_yaxis()
    axa.set_xlabel("Agent solve wall time (h)")
    axa.set_title("Agent axis", fontweight="bold")
    axa.legend(loc="lower right", frameon=False)
    clean_axes(axa, grid_axis="x")
    panel_label(axa, "(a)")

    axb.barh(
        y,
        [num(row, "total_tokens_m") for row in agent_rows],
        color=CACHE,
        edgecolor="#222222",
        linewidth=0.7,
        zorder=2,
    )
    axb.set_yticks(y, [r["short_label"] for r in agent_rows])
    axb.invert_yaxis()
    axb.set_xlabel("Total solving-side tokens (million)")
    clean_axes(axb, grid_axis="x")
    panel_label(axb, "(b)")

    agent_label_positions = {
        "gpt55": (6.2, 1.98),
        "gpt55_checklist": (9.0, 1.10),
        "opus48": (22.7, 1.82),
        "glm52": (34.0, 2.50),
        "sonnet46": (32.0, 1.72),
        "sol56_high": (11.5, 2.74),
        "sol56_ultra": (6.0, 3.10),
        "terra56_high": (26.5, 1.47),
        "luna56_high": (25.5, 0.43),
        "deepseek_v4_flash_high": (45.0, 0.35),
        "grok45_high": (17.0, 0.79),
        "deepseek_v4_pro_high": (60.5, 0.60),
        "grok46_high": (25.0, 2.30),
        "astra_high": (0.8, 0.80),
    }
    for row in agent_rows:
        x = num(row, "wall_total_sec") / 60 / int(row["passes"])
        yy = num(row, "cost_usd") / int(row["passes"])
        if np.isnan(yy):
            continue
        size = 30 + 11 * num(row, "cost_usd")
        color = COLORS[row["key"]]
        axc.scatter(x, yy, s=size, marker="o", facecolor=color, edgecolor="#222222", linewidth=0.8, alpha=0.95, zorder=3)
        label_x, label_y = agent_label_positions[row["key"]]
        axc.annotate(
            f"{row['short_label']}\n({row['passes']}/12)",
            (x, yy),
            xytext=(label_x, label_y),
            textcoords="data",
            fontsize=6.4,
            fontweight="bold",
            color=color,
            va="center",
        )
    axc.set_xlim(0, 76)
    axc.set_ylim(0, 3.35)
    axc.set_xlabel("Solve time per valid solution (min)")
    axc.set_ylabel("Solver cost per valid solution (USD)")
    axc.text(0.98, 0.97, "Marker area scales with total solver cost", transform=axc.transAxes, ha="right", va="top", fontsize=7.2, color="#444444")
    clean_axes(axc)
    panel_label(axc, "(c)")

    # Framework axis: unchanged original-paper comparison.
    fy = np.arange(len(framework_rows))
    fpass = np.array([num(r, "wall_passed_sec") / 3600 for r in framework_rows])
    ffail = np.array([num(r, "wall_failed_sec") / 3600 for r in framework_rows])
    axd.barh(fy, fpass, color=PASS, edgecolor="#222222", linewidth=0.7, zorder=2)
    axd.barh(fy, ffail, left=fpass, color=FAIL, edgecolor="#222222", linewidth=0.7, zorder=2)
    axd.set_yticks(fy, [r["short_label"] for r in framework_rows])
    axd.invert_yaxis()
    axd.set_xlabel("Agent solve wall time (h)")
    axd.set_title("Framework axis", fontweight="bold")
    clean_axes(axd, grid_axis="x")
    panel_label(axd, "(d)")

    axe.barh(fy, [num(r, "total_tokens_m") for r in framework_rows], color=CACHE, edgecolor="#222222", linewidth=0.7, zorder=2)
    axe.set_yticks(fy, [r["short_label"] for r in framework_rows])
    axe.invert_yaxis()
    axe.set_xlabel("Total solving-side tokens (million)")
    clean_axes(axe, grid_axis="x")
    panel_label(axe, "(e)")

    foffsets = {"tensorcircuit": (8, 6), "pennylane": (8, 5), "torchquantum": (-48, -13), "mindquantum": (-48, 8)}
    for row in framework_rows:
        x = num(row, "wall_total_sec") / 60 / int(row["passes"])
        yy = num(row, "cost_usd") / int(row["passes"])
        size = 30 + 11 * num(row, "cost_usd")
        color = COLORS[row["key"]]
        axf.scatter(x, yy, s=size, facecolor=color, edgecolor="#222222", linewidth=0.8, zorder=3)
        dx, dy = foffsets[row["key"]]
        axf.annotate(f"{row['short_label']}\n({row['passes']}/12)", (x, yy), xytext=(dx, dy), textcoords="offset points", fontsize=7.2, fontweight="bold", color=color, va="center")
    axf.set_xlim(7, 45)
    axf.set_ylim(0.8, 8.1)
    axf.set_xlabel("Solve time per valid solution (min)")
    axf.set_ylabel("Recorded cost per valid solution (USD)")
    clean_axes(axf)
    panel_label(axf, "(f)")

    fig.suptitle("Updated ORBIT-Q benchmark resource use", fontsize=13, fontweight="bold", y=0.995)
    save_all(fig, "fig4_updated_agent_framework_resources")


def make_expert_optimization() -> None:
    rows = read_csv("expert_optimization.csv")
    tasks = [r["task"] for r in rows]
    baseline = np.array([num(r, "baseline_sec") for r in rows])
    optimized = np.array([num(r, "optimized_sec") for r in rows])
    speedup = np.array([num(r, "speedup") for r in rows])
    short = {
        "01": "batched gate construction",
        "02": "batched exact purity",
        "03": "exact product-state reduction",
        "04": "batched probe networks",
        "05": "tuned OMECo path search",
        "06": "TC-native jaxode",
        "07": "exact ancilla/branch reduction",
        "08": "bounded mapped sampling",
        "09": "causal-cone pruning",
        "10": "fixed contraction program",
        "11": "layer and onsite fusion",
        "12": "batched Padé SU4",
    }

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.2, 5.25), gridspec_kw={"width_ratios": [1.08, 1.35], "wspace": 0.35})
    x = np.arange(len(tasks))
    width = 0.38
    axa.bar(x - width / 2, baseline, width, color="#7A7A7A", edgecolor="#222222", linewidth=0.6, label="Original expert", zorder=2)
    axa.bar(x + width / 2, optimized, width, color=PASS, edgecolor="#222222", linewidth=0.6, label="Human + AI", zorder=2)
    axa.set_yscale("log")
    axa.set_xticks(x, tasks)
    axa.set_xlabel("Challenge")
    axa.set_ylabel("End-to-end runtime (s, log scale)")
    axa.legend(frameon=False, loc="upper right")
    clean_axes(axa, grid_axis="y")
    panel_label(axa, "(a)")

    y = np.arange(len(tasks))
    bar_colors = ["#D55E00" if t in {"03", "07"} else PASS for t in tasks]
    bars = axb.barh(y, speedup, color=bar_colors, edgecolor="#222222", linewidth=0.75, zorder=2)
    axb.axvline(1.0, color="#777777", ls=(0, (3, 3)), lw=0.9)
    axb.set_xscale("log")
    axb.set_xlim(0.9, 78)
    axb.set_yticks(y, [f"Task {t}" for t in tasks])
    axb.invert_yaxis()
    axb.set_xlabel("End-to-end speedup (×, log scale)")
    clean_axes(axb, grid_axis="x")
    panel_label(axb, "(b)")
    for i, (t, value) in enumerate(zip(tasks, speedup)):
        axb.text(value * 1.06, i, f"{value:.2f}×  {short[t]}", va="center", ha="left", fontsize=7.2, color="#333333")

    axb.legend(
        handles=[
            Patch(facecolor=PASS, edgecolor="#222222", label="Framework-native optimization"),
            Patch(facecolor="#D55E00", edgecolor="#222222", label="Exact task reduction"),
        ],
        loc="lower right",
        frameon=False,
    )
    fig.suptitle("Human-expert implementations after AI-assisted optimization", fontsize=12.5, fontweight="bold", y=0.995)
    save_all(fig, "expert_optimization_updated_overview")


def make_astra_paired() -> None:
    """Use eleven measured pairs plus the reviewer-approved Task 08 estimate."""
    rows = read_csv("astra_expert_runtime.csv")
    tasks = [r["task"] for r in rows]
    expert = np.array([num(r, "expert_reference_sec") for r in rows])
    astra = np.array([num(r, "astra_sec") for r in rows])
    speedups = np.array([num(r, "expert_over_astra") for r in rows])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.2, 5.25),
                                gridspec_kw={"width_ratios": [1.08, 1.35], "wspace": 0.35})
    x = np.arange(len(rows))
    width = 0.38
    purple = COLORS["astra_high"]
    ax.bar(x-width/2, expert, width, color="#7A7A7A", edgecolor="#222222",
           linewidth=0.6, label="Expert reference", zorder=2)
    ax.bar(x+width/2, astra, width, color=purple, edgecolor="#222222",
           linewidth=0.6, label="Astra high", zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(1, 400)
    ax.set_xticks(x, tasks)
    ax.set_xlabel("Challenge")
    ax.set_ylabel("End-to-end runtime (s, log scale)")
    ax.legend(frameon=False, loc="upper right")
    for i, r in enumerate(rows):
        if np.isnan(expert[i]):
            ax.text(i-width/2, 1.16, r["expert_status"], color="#7A7A7A",
                    fontsize=6.8, ha="center", va="bottom", rotation=90)
    bx.barh(x, speedups, color=purple, edgecolor="#222222", linewidth=0.75, zorder=2)
    bx.axvline(1, color="#777777", ls=(0,(3,3)), lw=0.9)
    bx.set_xscale("log")
    bx.set_xlim(0.1, 35)
    bx.set_yticks(x, [f"Task {t}" for t in tasks])
    bx.invert_yaxis()
    bx.set_xlabel("End-to-end speedup (expert / Astra, ×, log scale)")
    for i, value in enumerate(speedups):
        if np.isnan(value):
            bx.text(1.08, i, f"Expert {rows[i]['expert_status']}", color="#555555", va="center", fontsize=7.2)
        else:
            bx.text(value*1.08, i, f"{value:.2f}×", va="center", ha="left", fontsize=7.2, color="#333333")
    for panel, grid_axis, label in [(ax,"y","(a)"),(bx,"x","(b)")]:
        clean_axes(panel, grid_axis=grid_axis)
        panel.set_axisbelow(True)
        panel_label(panel, label)
    fig.suptitle("Astra artifact runtime relative to the expert reference",
                 fontsize=12.5, fontweight="bold", y=0.995)
    save_all(fig, "astra_expert_paired_runtime")


def main() -> None:
    agents = read_csv("paper_agent_axis.csv")
    frameworks = read_csv("paper_framework_axis.csv")
    validate_new_slowdowns(agents)
    make_fig1c(agents, frameworks)
    make_fig2b(agents)
    make_fig4(agents, frameworks)
    make_astra_paired()
    # The earlier expert-optimization measurements and their figure are unchanged.
    # Call make_expert_optimization() explicitly to reproduce that archived figure.


if __name__ == "__main__":
    main()
