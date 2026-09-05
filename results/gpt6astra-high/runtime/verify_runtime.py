"""Validate the runtime supplement without executing any benchmark."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-6), message)


def verify(root=ROOT):
    root = Path(root)
    info = json.loads((root / "astra_expert_adjustment.json").read_text())
    raw_path = root / info["measured_source"]
    require(hashlib.sha256(raw_path.read_bytes()).hexdigest() == info["measured_source_sha256"],
            "Original measured source hash mismatch")
    with raw_path.open(newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    with (root / info["runtime_table"]).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for records in (raw_rows, rows):
        require(len(records) == 12 and {int(r["task"]) for r in records} == set(range(1, 13)),
                "Twelve unique tasks required")
    raw = {int(r["task"]): r for r in raw_rows}
    estimate = info["task08"]
    require(estimate["basis"] == "estimated" and estimate["original_expert_status"] == "OOM",
            "Task 08 must remain an estimate")
    close(estimate["expert_reference_sec"], 50, "Approved Task 08 reference")
    close(estimate["reviewer_proposed_sec"], estimate["published_reference_sec"] * estimate["reviewer_scale_factor"],
          "Reviewer scaling")
    require(raw[8]["expert_sec"] == "" and raw[8]["expert_status"] == "OOM"
            and raw[8]["included_in_aggregate"] == "0", "Preserve the incomplete expert measurement")
    expert_total, astra_total, ratios, faster = 0.0, 0.0, [], []
    for row in rows:
        task = int(row["task"])
        measured = raw[task]
        require(row["expert_reference_basis"] == ("estimated" if task == 8 else "measured"),
                "Only Task 08 may be estimated")
        if task != 8:
            require(measured["expert_status"] == "PASS" and measured["included_in_aggregate"] == "1",
                    "Measured expert pair must have completed")
        require(measured["astra_status"] == "PASS", "Astra functional status")
        expert = 50.0 if task == 8 else float(measured["expert_sec"])
        astra = float(measured["astra_sec"])
        close(row["expert_reference_sec"], expert, "Expert reference mismatch")
        close(row["astra_sec"], astra, "Astra measurement mismatch")
        close(row["astra_over_expert"], astra / expert, "Runtime ratio mismatch")
        close(row["expert_over_astra"], expert / astra, "Reciprocal speedup mismatch")
        require(row["source"] == (estimate["source"] if task == 8 else measured["source"]),
                "Runtime source provenance")
        expert_total += expert
        astra_total += astra
        ratios.append(astra / expert)
        if astra < expert:
            faster.append(task)
    gm = math.exp(sum(map(math.log, ratios)) / 12)
    summary = info["summary"]
    expected = dict(task_count=12, measured_pair_count=11, estimated_reference_count=1,
                    expert_sum_sec=expert_total, astra_sum_sec=astra_total,
                    ratio_of_sums=astra_total/expert_total, geometric_mean_ratio=gm,
                    geometric_mean_speedup=1/gm)
    for key, value in expected.items():
        close(summary[key], value, f"Runtime summary mismatch: {key}")
    require(summary["astra_faster_tasks"] == faster, "Faster task list mismatch")
    for stem in ("astra_expert_paired_runtime", "fig2b_updated_agent_axis"):
        require((root / "figs" / f"{stem}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), "Runtime PNG")
        require((root / "figs" / f"{stem}.pdf").read_bytes().startswith(b"%PDF-"), "Runtime PDF")
    return expected


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
