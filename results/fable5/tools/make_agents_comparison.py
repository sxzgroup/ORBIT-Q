"""Three-way agent comparison figure in the style of the upstream PR #5
submission figure: GPT-5.5 high (paper run) vs GPT-5.6 Sol high vs Fable 5.

Panel (a): agent-axis validity (failure rate) x artifact efficiency
(passed-task geometric-mean slowdown vs same-run expert reference).
Panel (b): task-level runtime ratios; F marks a failed reward.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "agents-comparison.json").read_text())
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)

runs = [
    (DATA["gpt55_paper_run"], "#0072B2", "GPT-5.5 high"),
    (DATA["gpt56_run"], "#D55E00", "GPT-5.6 Sol high"),
    (DATA["fable5_run"], "#2a9d8f", "Fable 5 (Cursor)"),
]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.8, 5.2))

# (a) validity x efficiency
ax0.scatter([0], [1], marker="D", s=170, facecolor="#eeeeee", edgecolor="#777777",
            linewidth=1.2, zorder=3)
ax0.annotate("Expert TC\n(12/12)", (0, 1), xytext=(7, -2), textcoords="offset points")
offsets = [(10, 6), (10, -34), (10, 6)]
for (run, color, label), off in zip(runs, offsets):
    x, y = run["failure_rate_pct"], run["passed_task_geometric_mean_slowdown"]
    ax0.scatter([x], [y], s=230, color=color, edgecolor="black", linewidth=1, zorder=4)
    ax0.annotate(f"{label}\n({run['passes']}/12, {y:.2f}x)", (x, y), xytext=off,
                 textcoords="offset points", color=color, weight="bold")
ax0.axhline(1, color="#888888", linestyle="--", linewidth=1)
ax0.axvline(0, color="#888888", linestyle="--", linewidth=1)
ax0.set_xlim(-2, 35)
ax0.set_ylim(0.8, 2.9)
ax0.set_xticks([0, 10, 20, 30])
ax0.set_xlabel("Failure rate (%)")
ax0.set_ylabel("Geometric mean runtime / expert TC\n(passed tasks; lower is better)")
ax0.set_title("(a) Agent-axis validity and artifact efficiency", loc="left")
ax0.grid(alpha=0.25)

# (b) task-level ratios
tasks = np.arange(1, 13)
width = 0.27
for k, (run, color, label) in enumerate(runs):
    tmap = {t["task"]: t for t in run["tasks"]}
    passed_key = "passed" if "passed" in run["tasks"][0] else "reward"
    def ok(t):
        return t.get("passed", t.get("reward") == 1)
    vals = [tmap[i]["runtime_ratio"] if ok(tmap[i]) and tmap[i]["runtime_ratio"] else np.nan
            for i in tasks]
    ax1.bar(tasks + (k - 1) * width, vals, width, label=label, color=color,
            edgecolor="black", linewidth=0.5)
    for i in tasks:
        if not ok(tmap[i]):
            ax1.text(i + (k - 1) * width, 0.40, "F", ha="center", va="center",
                     color=color, weight="bold", fontsize=9)
ax1.axhline(1, color="#888888", linestyle="--", linewidth=1)
ax1.set_yscale("log")
ax1.set_ylim(0.33, 22)
ax1.set_xticks(tasks)
ax1.set_xticklabels([f"{x:02d}" for x in tasks])
ax1.set_xlabel("Challenge task")
ax1.set_ylabel("Candidate runtime / same-run expert reference")
ax1.set_title("(b) Task-level artifact runtime; F = failed reward", loc="left")
ax1.legend(frameon=False, loc="upper left")
ax1.grid(alpha=0.25, axis="y", which="both")

fig.suptitle("ORBIT-Q TensorCircuit agent comparison: GPT-5.5 high vs GPT-5.6 Sol high vs Fable 5",
             y=1.01, fontsize=12)
fig.tight_layout()
out = FIGS / "fable5-agents-comparison.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(out)
