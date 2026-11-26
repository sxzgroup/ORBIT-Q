"""Rebuild this draft's five charts from its frozen JSON data. Requires matplotlib and numpy."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np

LABELS = {
    "sol": "6.2 Sol high",
    "astra": "6.2 Astra high",
    "terra": "6.1 Terra high",
    "neon": "Gemini 4 Neon high",
    "carbon": "Gemini 4 Carbon high",
    "hydrogen": "Gemini 4 Hydrogen high",
}
FAMILY = {
    "sol": "#004B85",
    "astra": "#6932A0",
    "terra": "#00784F",
    "neon": "#E69F00",
    "carbon": "#0072B2",
    "hydrogen": "#CC3311",
}
# Exact names in the two frozen input tables; do not filter whole model families.
RETIRED = {
    "5.6 Sol high": "Sol",
    "5.6 Sol ultra": "Sol ultra",
    "5.6 Terra high": "Terra",
    "5.6 Luna high": "Luna",
    "Grok 4.5 high": "Grok 4.5",
}
plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "svg.hashsalt": "orbitq-agy",
    }
)


def solver_label(v: str) -> str:
    return LABELS[v]


def save(fig, dest: Path, name: str) -> None:
    figs = dest / "figures"
    figs.mkdir(exist_ok=True)
    for fmt in ("png", "svg", "pdf"):
        opts = {"svg": {"Date": None}, "pdf": {"CreationDate": None, "ModDate": None}}
        fig.savefig(
            figs / f"{name}.{fmt}",
            dpi=200,
            metadata=opts.get(fmt, {}),
            facecolor="white",
        )
        if fmt == "svg":
            path = figs / f"{name}.{fmt}"
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


def outcomes(runs, out: Path) -> None:
    x = np.arange(1, 13)
    fig, ax = plt.subplots(figsize=(9.6, 0.38 * len(runs) + 1.9), layout="constrained")
    m = np.array([[n in r["passes"] for n in x] for r in runs])
    ax.imshow(m, cmap=ListedColormap(["#F7F7F7", "#D9ECE7"]), vmin=0, vmax=1, aspect="auto")
    for i, row in enumerate(m):
        for j, p in enumerate(row):
            ax.text(
                j,
                i,
                "P" if p else "F",
                ha="center",
                va="center",
                color="#175B48" if p else "#444444",
                weight="bold",
            )
    ax.set(
        xticks=x - 1,
        xticklabels=[f"{n:02d}" for n in x],
        yticks=range(len(runs)),
        yticklabels=[f"{r['label']}   {len(r['passes'])}/12" for r in runs],
        xlabel="Challenge",
        title="TensorCircuit task outcomes",
    )
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(runs), 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="both", length=0)
    save(fig, out, "task-outcomes")


def efficiency(ax, runs):
    filtered = [r for r in runs if len(r["passes"]) and r["total_cost_usd"] is not None]
    offsets = {
        "Gemini 4 Neon high": (-11, 5),
        "Gemini 4 Carbon high": (5, 5),
        "Gemini 4 Hydrogen high": (5, 4),
    }
    for i, r in enumerate(filtered, 1):
        p = len(r["passes"])
        total = r["total_cost_usd"]
        x = r["agent_wall_sec"] / p / 60
        y = total / p
        ax.scatter(
            x,
            y,
            s=total * 25,
            color=r["color"],
            edgecolors="black",
            linewidths=0.7,
            alpha=0.84,
            zorder=3,
        )
        dx, dy = offsets.get(r["label"], (3, 4))
        ax.annotate(
            str(i),
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7,
            weight="bold",
            zorder=4,
            bbox=dict(
                boxstyle="round,pad=.10", facecolor="white", edgecolor="none", alpha=0.8
            ),
        )
    ax.set(
        xlabel="Solver minutes per pass",
        ylabel="Solver USD per pass",
        title="(c)  Resource efficiency",
    )
    ax.set_xlim(
        0, max(74, max(r["agent_wall_sec"] / len(r["passes"]) / 60 for r in filtered) * 1.12)
    )
    ax.set_ylim(
        0, max(r["total_cost_usd"] / len(r["passes"]) for r in filtered) * 1.18
    )
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    return [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=6,
            color=r["color"],
            label=f"{i}  {r['label']}  ({len(r['passes'])}/12)",
        )
        for i, r in enumerate(filtered, 1)
    ]


def resources(v: str, rows, runs, out: Path) -> None:
    x = np.arange(1, 13)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.75))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.82, bottom=0.39, wspace=0.34)
    wall = np.array([r["solver_wall_sec"] / 60 for r in rows])
    axes[0].bar(x, wall, color="#009E73", edgecolor="black", lw=0.5)
    axes[0].set(
        title=f"(a)  Solver time · {wall.sum():.2f} min",
        ylabel="Solver active time (min)",
    )
    inp = np.array([r["n_input_tokens"] for r in rows], dtype=float)
    cache = np.array([r["n_cache_tokens"] for r in rows], dtype=float)
    output = np.array([r["n_output_tokens"] for r in rows], dtype=float)
    bottom = np.zeros(12)
    for arr, color, lab in [
        (inp - cache, "#595959", "Non-cache prompt"),
        (cache, "#56B4E9", "Cache-read prompt"),
        (output, "#D55E00", "Output"),
    ]:
        axes[1].bar(
            x, arr / 1e6, bottom=bottom, color=color, edgecolor="black", lw=0.4, label=lab
        )
        bottom += arr / 1e6
    axes[1].set(
        title=f"(b)  Solver tokens · {bottom.sum():.3f}M", ylabel="Tokens (million)"
    )
    axes[1].legend(fontsize=7, frameon=False)
    for a in axes[:2]:
        a.set(xticks=x, xticklabels=[f"{n:02d}" for n in x], xlabel="Challenge")
        a.tick_params(axis="x", labelsize=7)
        a.grid(axis="y", alpha=0.18)
        a.set_axisbelow(True)
    handles = efficiency(axes[2], runs)
    axes[2].set_box_aspect(1)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.058),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        f"{solver_label(v)} · agent-side resource use",
        x=0.055,
        ha="left",
        weight="bold",
        fontsize=15,
    )
    fig.text(
        0.055,
        0.024,
        "Marker area is proportional to total API-equivalent solver cost. Verifier resources excluded.",
        fontsize=8,
        color="#555555",
    )
    save(fig, out, "agent-resources")

    fig = plt.figure(figsize=(6.8, 6.8))
    ax = fig.add_axes([0.21, 0.38, 0.60, 0.56])
    ax.set_box_aspect(1)
    handles = efficiency(ax, runs)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.052),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.text(
        0.5,
        0.022,
        "Marker area is proportional to total API-equivalent solver cost.",
        ha="center",
        fontsize=8,
    )
    save(fig, out, "resource-efficiency-square")


def paired_fig(v: str, out: Path, pairs, raw_pairs) -> None:
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(11.8, 4.95), layout="constrained", gridspec_kw={"width_ratios": [1.05, 1.25]}
    )
    x = np.arange(1, 13)
    ex = np.array(
        [
            r["expert_reference_sec"]
            if r["expert_reference_basis"] == "measured" and r["expert_reference_sec"]
            else np.nan
            for r in pairs
        ]
    )
    model = np.array([r["model_sec"] or np.nan for r in pairs])
    speed = ex / model
    a.bar(x - 0.18, ex, 0.36, color="#808080", edgecolor="#333333", lw=0.5, label="Expert reference")
    estimated = pairs[7]["expert_reference_basis"] == "estimated"
    proxy = bool(estimated and np.isfinite(model[7]))
    if estimated:
        a.bar(
            8 - 0.18,
            pairs[7]["expert_reference_sec"],
            0.36,
            color="white",
            edgecolor="#646464",
            hatch="////",
            lw=0.65,
            label="Task 08: 50 s proxy",
        )
    a.bar(x + 0.18, model, 0.36, color=FAMILY[v], edgecolor="#333333", lw=0.5, label=solver_label(v))
    a.set_yscale("log")
    a.set(
        ylim=(min(0.8, np.nanmin([ex, model]) * 0.8), np.nanmax([ex, model]) * 1.65),
        xticks=x,
        xticklabels=[f"{i:02d}" for i in x],
        xlabel="Challenge",
        ylabel="End-to-end runtime (s, log scale)",
        title="(a)  Artifact execution time",
    )
    a.legend(frameon=False, fontsize=8)
    b.set_xscale("log")
    b.barh(x, speed, height=0.62, color=FAMILY[v], edgecolor="#333333", lw=0.6)
    est = 0.0
    if proxy:
        est = pairs[7]["expert_reference_sec"] / model[7]
        b.barh(8, est, height=0.62, color="white", edgecolor=FAMILY[v], hatch="////", lw=0.8)
    b.axvline(1, color="#777777", ls="--", lw=1)
    b.set(
        xlim=(
            min(0.1, np.nanmin(speed) * 0.7),
            max(3.0, np.nanmax(speed) * 2.2, est * 2.2 if proxy else 0.0),
        ),
        yticks=x,
        yticklabels=[f"Task {i:02d}" for i in x],
        xlabel="Speedup (expert / artifact, ×, log scale)",
        title=f"(b)  Speedup · {int(np.isfinite(speed).sum()) + int(proxy)} valid ratios",
    )
    b.invert_yaxis()
    for j, z in enumerate(speed):
        if proxy and j == 7:
            b.text(est * 1.07, 8, f"{est:.2f}× (proxy)", va="center", fontsize=8)
            continue
        t = (
            f"{z:.2f}×"
            if np.isfinite(z)
            else ("No valid submission" if not raw_pairs[j]["candidate_valid"] else "No measured pair")
        )
        b.text(z * 1.07 if np.isfinite(z) else 1.08, j + 1, t, va="center", fontsize=8)
    for ax, axis in [(a, "y"), (b, "x")]:
        ax.grid(axis=axis, alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle(f"{solver_label(v)} · artifact runtime relative to the expert reference", fontweight="bold")
    save(fig, out, "artifact-vs-expert")


def slowdowns(dest: Path, data) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2), layout="constrained")
    ax.axhline(1, ls="--", color="#888888", lw=0.8)
    ax.axvline(0, ls="--", color="#888888", lw=0.8)
    ax.scatter(
        0,
        1,
        marker="D",
        s=85,
        facecolors="none",
        edgecolors="#555555",
        label="Expert TC reference",
        zorder=3,
    )
    for i, (label, x, y) in enumerate(data["historical"]):
        if label in RETIRED.values():
            continue
        ax.scatter(
            x,
            y,
            s=36,
            color=("#7B2CBF" if label == "Astra" else plt.cm.tab20(i / 19)),
            edgecolors="#555555",
            lw=0.6,
            label=("6 Astra" if label == "Astra" else label),
            zorder=3,
        )
    for row in data["new"]:
        if row["ratio"] is not None:
            ax.scatter(
                row["failure_pct"],
                row["ratio"],
                marker=row["marker"],
                s=84,
                alpha=0.88,
                color=FAMILY[row["variant"]],
                edgecolors="black",
                lw=0.7,
                label=f"{row['label']} ({row['pairs']} pairs)",
                zorder=5 if row["variant"] == "terra" else 4,
            )
    ax.set(
        xlim=(-8, max(68, ax.get_xlim()[1])),
        ylim=(-0.45, max(8.3, ax.get_ylim()[1])),
        xlabel="Failure rate (%)",
        ylabel="Artifact time / expert time (geometric mean)",
        title="Validity and artifact efficiency",
    )
    zoom = ax.inset_axes([0.07, 0.49, 0.43, 0.43])
    zoom.set_facecolor("#FAFAFA")
    zoom.axhline(1, ls=":", color="#AAAAAA", lw=0.7)
    zoom.axvline(0, ls=":", color="#AAAAAA", lw=0.7)
    beyond = [
        ("Expert", 0.0, 1.0, "D", "#888888"),
    ]
    for label, x, y in data["historical"]:
        if label in RETIRED.values():
            continue
        if y <= 1.0:
            beyond.append(
                ("6 Astra" if label == "Astra" else label, x, y, "o", "#7B2CBF" if label == "Astra" else "#555555")
            )
    for r in data["new"]:
        if r["ratio"] is not None and r["ratio"] <= 1.0:
            beyond.append(
                (r["label"].removesuffix(" high"), r["failure_pct"], r["ratio"], r["marker"], FAMILY[r["variant"]])
            )
    zoom_offsets = {
        "Expert": (5, 0),
        "6 Astra": (5, 0),
        "6.2 Sol": (5, 0),
        "6.2 Astra": (5, 0),
        "6.1 Terra": (5, 0),
        "Gemini 4 Neon": (5, 0),
        "Gemini 4 Carbon": (5, -1),
    }
    for name, zx, zy, m, c in beyond:
        zoom.scatter(
            zx,
            zy,
            s=28,
            marker=m,
            facecolors="none" if name == "Expert" else c,
            edgecolors=c if name == "Expert" else "black",
            lw=0.7,
            zorder=4,
        )
        dx, dy = zoom_offsets.get(name, (5, 0))
        zoom.annotate(
            name,
            (zx, zy),
            xytext=(dx, dy),
            textcoords="offset points",
            va="center",
            fontsize=6.8,
            color="#2B2B2B",
            zorder=5,
        )
    zoom.set(xlim=(-2.0, 28.0), ylim=(0.05, 1.14), xticks=[0, 10, 20], yticks=[0.2, 0.6, 1.0])
    zoom.set_xlabel("Failure rate (%)", fontsize=6.5, labelpad=1)
    zoom.tick_params(labelsize=6, length=2)
    zoom.set_title("Beyond expert zone", fontsize=7.4, pad=3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    save(fig, dest, "artifact-efficiency")


def main() -> None:
    out = Path(__file__).resolve().parent
    read = lambda name: json.loads((out / name).read_text())
    v = read("execution-protocol.json")["solver"].split("-")[2]
    rows = read("trials.json")
    runs = [r for r in read("resources-comparison.json") if r["label"] not in RETIRED]
    outcomes(runs, out)
    resources(v, rows, runs, out)
    paired_fig(v, out, read("published-runtimes.json"), read("paired-runtimes.json"))
    slowdowns(out, read("runtime-comparison.json"))


if __name__ == "__main__":
    main()
