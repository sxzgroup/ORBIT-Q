"""Generate the two paper-style summary figures (mirroring the upstream
README "Main Results" pair: one artifact-efficiency figure, one resource-use
figure) from the recorded per-task results.

Reads challenge-NN/reward.json (official stamps),
challenge-NN/runtime-comparison.json, and agent_solve_time.json, writes:
  figs/fable5_ratio_bars.png         - artifact efficiency vs expert reference
  figs/fable5_agent_resource_use.png - agent solve time + artifact runtime
Run from the repository root:  python3 results/fable5/tools/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)

tasks, official, matched, needs_c128 = [], [], [], []
cand_t, ref_t, stamp_t = [], [], []
for nn in range(1, 13):
    d = ROOT / f"challenge-{nn:02d}"
    rc = json.loads((d / "runtime-comparison.json").read_text())
    rw = json.loads((d / "reward.json").read_text())
    key = "runtime_ratio_candidate_over_reference"
    off = rc.get(key) or rc["primary"][key]
    cand = rc.get("candidate_runtime_sec") or rc["primary"]["candidate_runtime_sec"]
    ref = (rc.get("reference_official") or rc["primary"]["reference_official"])["runtime_sec"]
    mc = rc.get("matched_precision_control")
    if mc and mc.get("matched_precision_ratio"):
        m, req128 = mc["matched_precision_ratio"], False
    elif mc:
        m, req128 = off, True
    else:
        m, req128 = off, False
    tasks.append(f"{nn:02d}")
    official.append(off)
    matched.append(m)
    needs_c128.append(req128)
    cand_t.append(cand)
    ref_t.append(ref)
    stamp_t.append(rw["runtime_sec"])

solve = json.loads((ROOT / "agent_solve_time.json").read_text())["solve_wall_time_min"]
solve_min = [solve[t] for t in tasks]

official = np.array(official)
matched = np.array(matched)

# --- Figure 1: per-task artifact efficiency (official bars + c64 controls) ---
fig, ax = plt.subplots(figsize=(8.4, 4.4))
bar_colors = ["#2a9d8f" if r <= 1.0 else "#e9c46a" if r <= 4 else "#e76f51" for r in official]
bars = ax.bar(tasks, official, color=bar_colors, edgecolor="black", linewidth=0.5)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
for b, r, ct, rt in zip(bars, official, cand_t, ref_t):
    ax.annotate(f"{r:.2f}\n{ct:.0f}s/{rt:.0f}s", (b.get_x() + b.get_width() / 2, r),
                textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7.5)
for i, (m, off, req) in enumerate(zip(matched, official, needs_c128)):
    if req:
        ax.annotate("c64 fails", (i, 0.47), ha="center", fontsize=8,
                    color="white", fontweight="bold")
    elif abs(m - off) > 1e-9:
        ax.scatter([i], [m], marker="D", s=42, color="white",
                   edgecolor="#264653", linewidth=1.4, zorder=4)
        ax.annotate(f"{m:.2f}", (i, m), textcoords="offset points",
                    xytext=(14, -3), fontsize=7.5, color="#264653", fontweight="bold")
ax.scatter([], [], marker="D", s=42, color="white", edgecolor="#264653",
           linewidth=1.4, label="complex64 control (matched precision)")
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
ax.set_yscale("log")
ax.set_ylim(0.4, 40)
ax.set_xlabel("challenge task")
ax.set_ylabel("T / T$_{ref}$  (log scale)")
ax.set_title("Per-task artifact efficiency vs reference, same machine & image "
             "(annotated: candidate/reference seconds)", fontsize=9.5)
ax.grid(alpha=0.25, axis="y", which="both")
fig.tight_layout()
fig.savefig(FIGS / "fable5_ratio_bars.png", dpi=180)

# --- Figure 2: agent-side solve cost + artifact runtime (fig4-style) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0))
ax1.bar(tasks, solve_min, color="#457b9d", edgecolor="black", linewidth=0.5)
for i, v in enumerate(solve_min):
    ax1.annotate(str(v), (i, v), textcoords="offset points", xytext=(0, 3),
                 ha="center", fontsize=8)
ax1.set_xlabel("challenge task")
ax1.set_ylabel("agent solve wall time (min)")
ax1.set_title("Agent-side cost: solve wall time per task\n"
              "(session-timeline reconstruction; includes prototyping and\n"
              "failed iterations; token/service cost not exposed by harness)",
              fontsize=9)
ax1.grid(alpha=0.25, axis="y")

ax2.bar(tasks, stamp_t, color="#8d99ae", edgecolor="black", linewidth=0.5,
        label="candidate (official stamp)")
ax2.scatter(tasks, ref_t, marker="_", s=220, color="#d62828", linewidth=2.5,
            label="expert reference (same machine)", zorder=4)
ax2.axhline(300, color="gray", linestyle=":", linewidth=1)
ax2.text(-0.45, 300, "300s budget ", fontsize=7.5, color="gray", va="bottom")
ax2.set_yscale("log")
ax2.set_xlabel("challenge task")
ax2.set_ylabel("artifact runtime (s, log)")
ax2.set_title("Artifact-side cost: evaluator-timed runtime\n"
              "(bars: stamped candidate; red dashes: expert reference)",
              fontsize=9)
ax2.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
ax2.grid(alpha=0.25, axis="y", which="both")
fig.tight_layout()
fig.savefig(FIGS / "fable5_agent_resource_use.png", dpi=180)

print("figures written:", sorted(p.name for p in FIGS.iterdir()))
print("total solve time:", sum(solve_min), "min")
