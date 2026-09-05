#!/usr/bin/env python3
"""Validate paper tables without plotting dependencies or network access."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

DATA = Path(__file__).resolve().parent / "source_data"
EXPECTED = dict(zip([
    "gpt56_sol_high", "gpt56_sol_ultra", "gpt56_terra_high", "gpt56_luna_high",
    "deepseek_v4_flash_high", "grok45_high", "deepseek_v4_pro_high",
    "grok46_high", "astra_high"], [10, 10, 9, 9, 5, 8, 5, 9, 12]))
AGENT_KEYS = dict(zip(list(EXPECTED)[:4], [
    "sol56_high", "sol56_ultra", "terra56_high", "luna56_high"]))


def close(actual, expected, label, tolerance=5.1e-7):
    if not math.isclose(float(actual), expected, rel_tol=0, abs_tol=tolerance):
        raise ValueError(f"{label}: {actual} != {expected}")


def require(condition, label):
    if not condition:
        raise ValueError(label)


def load(data=DATA):
    result = {}
    for name in ("paper_agent_axis", "benchmark_models", "task_outcomes",
                 "paper_expert_runtimes", "astra_expert_paired", "astra_expert_runtime"):
        with (data / f"{name}.csv").open(newline="") as handle:
            result[name] = list(csv.DictReader(handle))
    result["evidence"] = json.loads((data / "recent_pr_evidence.json").read_text())
    result["astra_adjustment"] = json.loads((data / "astra_expert_adjustment.json").read_text())
    require(hashlib.sha256((data / "astra_expert_paired.csv").read_bytes()).hexdigest()
            == result["astra_adjustment"]["measured_source_sha256"], "Raw paired source hash")
    return result


def validate(tables, check_git_sources=False):
    agent_rows = tables["paper_agent_axis"]
    agents = {r["key"]: r for r in agent_rows}
    models = {r["model"]: r for r in tables["benchmark_models"]}
    tasks = tables["task_outcomes"]
    require(len(agent_rows) == len(agents) == 14, "14 unique agent rows required")
    require(len(tables["benchmark_models"]) == len(models) == 9, "9 models required")
    require(set(models) == set(EXPECTED), "Unexpected model keys")
    require(len(tasks) == len({(r["model"], r["task"]) for r in tasks}) == 108,
            "108 unique task rows required")
    refs = {int(r["task"]): float(r["expert_tc_runtime_sec"])
            for r in tables["paper_expert_runtimes"]}
    require(set(refs) == set(range(1, 13)), "12 expert references required")
    for model, passes in EXPECTED.items():
        rows = [r for r in tasks if r["model"] == model]
        require({int(r["task"]) for r in rows} == set(range(1, 13)), model)
        require(all(r["final_pass"] in ("0", "1") for r in rows), model)
        close(sum(int(r["final_pass"]) for r in rows), passes, model)
        axis = agents[AGENT_KEYS.get(model, model)]
        for aggregate in (models[model], axis):
            close(aggregate["passes"], passes, model)
            close(aggregate["failures"], 12-passes, model)
        ratios = [float(r["runtime_sec"])/refs[int(r["task"])]
                  for r in rows if r["final_pass"] == "1"]
        require(all(v > 0 for v in ratios), f"{model}: positive passed runtimes")
        gm = math.exp(sum(map(math.log, ratios))/len(ratios))
        close(axis["gm_slowdown"], gm, model)
        close(models[model]["gm_slowdown"], gm, model, tolerance=0.00051)
        close(axis["wall_total_sec"], float(axis["wall_passed_sec"])
              + float(axis["wall_failed_sec"]), model, tolerance=0.001)
        task08 = next(r for r in rows if int(r["task"]) == 8)
        require(task08["final_pass"] == ("1" if model == "astra_high" else "0"),
                f"{model}: preserve Task 08 source outcome")

    evidence = tables["evidence"]
    require({m["key"] for m in evidence["models"]} == set(list(EXPECTED)[-3:]),
            "Expected evidence for three recent campaigns")
    for model in evidence["models"]:
        key, rows = model["key"], model["tasks"]
        require(len(rows) == 12, f"{key}: evidence task count")
        source = model["source"]
        if check_git_sources:
            raw = subprocess.check_output(["git", "show", f"{source['commit']}:{source['path']}"],
                                          cwd=DATA.parent.parent)
            require(hashlib.sha256(raw).hexdigest() == source["sha256"],
                    f"{key}: source SHA-256 mismatch")
            if key == "grok46_high":
                close(agents[key]["cost_usd"], json.loads(raw)["agent_usage"]["list_price_cost_usd"], key)
        indexed = {int(r["task"]): r for r in tasks if r["model"] == key}
        for record in rows:
            row = indexed[record["task"]]
            for field in ("raw_reward", "final_pass", "functional", "static", "audit", "runtime_sec"):
                if record[field] is None:
                    require(row[field] == "", f"{key}: missing {field} must stay missing")
                else:
                    close(row[field], record[field], f"{key}/{record['task']}/{field}")
        wall = sum(r["agent_wall_sec"] for r in rows)
        passed_wall = sum(r["agent_wall_sec"] for r in rows if r["final_pass"])
        prompt, cache, output = [sum(r[field] for r in rows)
                                for field in ("input_tokens", "cache_tokens", "output_tokens")]
        # Archived Grok 4.6 accounting uses these recorded per-million rates.
        cost = ((prompt-cache)*2 + cache*0.5 + output*6)/1e6 if key == "grok46_high" else sum(r["cost_usd"] for r in rows)
        values = {"wall_total_sec": wall, "wall_passed_sec": passed_wall,
                  "wall_failed_sec": wall-passed_wall, "total_tokens_m": (prompt+output)/1e6,
                  "noncache_prompt_m": (prompt-cache)/1e6, "cache_prompt_m": cache/1e6,
                  "output_tokens_m": output/1e6, "cost_usd": cost}
        for field, value in values.items():
            close(agents[key][field], value, f"{key}/{field}")
        values = {"agent_wall_min": wall/60, "total_tokens_m": (prompt+output)/1e6,
                  "input_tokens_m": prompt/1e6, "cache_tokens_m": cache/1e6,
                  "output_tokens_m": output/1e6, "cost_usd": cost,
                  "cost_per_valid_usd": cost/EXPECTED[key],
                  "solve_time_per_valid_min": wall/60/EXPECTED[key]}
        for field, value in values.items():
            close(models[key][field], value, f"{key}/{field}")

    paired = tables["astra_expert_paired"]
    require(len(paired) == 12 and {int(r["task"]) for r in paired} == set(range(1, 13)),
            "12 paired records required")
    by_task = {r["task"]: r for r in evidence["paired_astra"]["tasks"]}
    valid = []
    for row in paired:
        record = by_task[int(row["task"])]
        for field in ("expert_status", "astra_status"):
            require(row[field] == record[field], f"Paired status: {field}")
        close(row["astra_sec"], record["astra_sec"], "Paired Astra runtime")
        if record["expert_sec"] is None:
            require(row["included_in_aggregate"] == "0" and
                    all(row[f] == "" for f in ("expert_sec", "astra_over_expert", "expert_over_astra")),
                    "OOM must remain missing and excluded")
        else:
            require(row["included_in_aggregate"] == "1", "Completed pair must be included")
            expert, astra = float(row["expert_sec"]), float(row["astra_sec"])
            close(expert, record["expert_sec"], "Paired expert runtime")
            close(row["astra_over_expert"], astra/expert, "Paired ratio")
            close(row["expert_over_astra"], expert/astra, "Paired speedup")
            valid.append((expert, astra))
    summary = evidence["paired_astra"]["summary"]
    close(len(valid), summary["paired_functional_passes"], "Paired count")
    close(sum(e for e, a in valid), summary["expert_sum_sec"], "Paired expert total")
    close(sum(a for e, a in valid), summary["astra_sum_sec"], "Paired Astra total")
    close(sum(a for e, a in valid)/sum(e for e, a in valid), summary["ratio_of_sums"], "Ratio of sums")
    gm = math.exp(sum(math.log(a/e) for e, a in valid)/len(valid))
    close(gm, summary["geometric_mean_ratio"], "Paired GM")
    close(sum(a < e for e, a in valid), len(summary["astra_faster_tasks"]), "Faster tasks")
    require(abs(gm-float(agents["astra_high"]["gm_slowdown"])) > 0.4,
            "Paired and fixed-paper baselines must not be mixed")

    adjusted = tables["astra_expert_runtime"]
    adjustment = tables["astra_adjustment"]
    require(len(adjusted) == 12 and {int(r["task"]) for r in adjusted} == set(range(1, 13)),
            "12 adjusted runtime records required")
    assumption = adjustment["task08"]
    close(assumption["expert_reference_sec"], 50, "Task 08 approved reference")
    close(assumption["reviewer_proposed_sec"],
          assumption["published_reference_sec"]*assumption["reviewer_scale_factor"],
          "Reviewer reference scaling")
    require(assumption["basis"] == "estimated" and assumption["original_expert_status"] == "OOM",
            "Task 08 estimate must not become a successful measurement")
    for row in adjusted:
        task = int(row["task"])
        record = by_task[task]
        require(row["expert_reference_basis"] == ("estimated" if task == 8 else "measured"),
                "Only Task 08 uses an estimated expert reference")
        close(row["astra_sec"], record["astra_sec"], "Adjusted Astra measurement")
        expert = 50 if task == 8 else record["expert_sec"]
        astra = record["astra_sec"]
        close(row["expert_reference_sec"], expert, "Adjusted expert reference")
        close(row["astra_over_expert"], astra/expert, "Adjusted ratio")
        close(row["expert_over_astra"], expert/astra, "Adjusted speedup")
        require(row["source"] == (assumption["source"] if task == 8 else evidence["paired_astra"]["source"]),
                "Adjusted source provenance")
    summary = adjustment["summary"]
    close(summary["task_count"], 12, "Adjusted count")
    close(summary["measured_pair_count"], 11, "Measured pair count")
    close(summary["estimated_reference_count"], 1, "Estimated count")
    expert_total = sum(float(r["expert_reference_sec"]) for r in adjusted)
    astra_total = sum(float(r["astra_sec"]) for r in adjusted)
    adjusted_gm = math.exp(sum(math.log(float(r["astra_sec"])/float(r["expert_reference_sec"]))
                               for r in adjusted)/len(adjusted))
    for field, value in {"expert_sum_sec": expert_total, "astra_sum_sec": astra_total,
                         "ratio_of_sums": astra_total/expert_total,
                         "geometric_mean_ratio": adjusted_gm,
                         "geometric_mean_speedup": 1/adjusted_gm}.items():
        close(summary[field], value, f"Adjusted {field}")
    require(summary["astra_faster_tasks"] == [int(r["task"]) for r in adjusted
            if float(r["astra_sec"]) < float(r["expert_reference_sec"])], "Adjusted faster tasks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-git-sources", action="store_true",
                        help="Also check pinned source objects when available locally")
    args = parser.parse_args()
    validate(load(), args.check_git_sources)
    print("PASS: 14 agent rows, 108 outcomes, three source campaigns; Astra has 11 measured pairs + one estimated reference")
