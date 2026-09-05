#!/usr/bin/env python3
"""Summarize frozen artifacts; historical data stays on the host."""
import ast
from datetime import datetime
import hashlib
import json
import math
import os
import re
from pathlib import Path
import subprocess

from run_benchmark import ROOT, OUT, JOBS, command, dump, BASE

HISTORY = {
    "sol_high": ("a395c1cd0444630f2244fb89e3bcecde667e3372", "gpt56sol-high", "gpt56_sol_high"),
    "sol_ultra": ("552481c7d133c1bc48e7b0e379cab8fcfaf3851b", "gpt56sol-ultra", "gpt56_sol_ultra"),
}


def read_git(ref, path):
    return json.loads(subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT, text=True))


def read_harbor_json(path):
    # Harbor redacts the literal env value of CODEX_FORCE_AUTH_JSON (true),
    # including boolean occurrences in archived metadata. Preserve originals;
    # treat redacted fields as unknown. Numeric reward/token fields are intact.
    raw = path.read_text()
    normalized, count = re.subn(r'(:\s*)\[REDACTED\](?=\s*[,}\]])', r'\1null', raw)
    data = json.loads(normalized)
    if count:
        original = path.with_suffix(".raw.txt")
        if original.exists():
            assert original.read_text() == raw
        else:
            original.write_text(raw)
        data["_archive_normalization"] = {
            "original_file": original.name, "redacted_values_represented_as_null": count,
            "note": "Harbor replaced literal true with an unquoted redaction marker. Raw bytes retained; redacted values are unknown. Numeric metrics unchanged."}
        dump(path, data)
    return data


def historical():
    models = {}
    for name, (ref, directory, key) in HISTORY.items():
        resources = read_git(ref, f"results/{directory}/agent-resource-use.json")[key]
        tasks = []
        for n in range(1, 13):
            folder = f"results/{directory}/challenge-{n:02d}"
            reward = read_git(ref, f"{folder}/reward.json")
            # Some historical archives contain unquoted redaction markers in
            # unrelated fields. Read the explicit numeric audit score only.
            audit_text = subprocess.check_output(["git", "show", f"{ref}:{folder}/audit-details.json"], cwd=ROOT, text=True)
            row = next(t for t in resources["tasks"] if t["task"] == n)
            matches = re.findall(r'"llm_audit_score"\s*:\s*([01](?:\.0)?)', audit_text)
            assert len(matches) == 1, (name, n, matches)
            raw_audit = float(matches[0])
            raw_reward = reward["functional_score"] * reward["static_policy_score"] * raw_audit
            tasks.append({"task": n, "raw_reward": raw_reward,
                          "final_valid": row["passed"], "runtime_sec": reward["runtime_sec"],
                          "agent_wall_sec": row["agent_wall_sec"], "total_tokens": row["total_tokens"],
                          "input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"],
                          "cache_tokens": row["cache_tokens"], "cost_usd": row.get("cost_usd")})
        models[name] = {"commit": ref, "source": f"results/{directory}",
                        "label": resources["label"], "model": resources["model"],
                        "effort": resources["reasoning_effort"], "tasks": tasks,
                        "raw_passes": sum(t["raw_reward"] == 1 for t in tasks),
                        "final_passes": sum(t["final_valid"] for t in tasks),
                        "agent_wall_sec": resources["agent_wall_sec"],
                        "input_tokens": resources["input_tokens"], "cache_tokens": resources["cache_tokens"],
                        "output_tokens": resources["output_tokens"], "total_tokens": resources["total_tokens"],
                        "cost_usd": resources["cost_usd"]}
    return models


def astra():
    tasks = []
    for n in range(1, 13):
        dest = OUT / f"challenge-{n:02d}"
        if not (dest / "status.json").exists():
            continue
        status = json.loads((dest / "status.json").read_text())
        result = read_harbor_json(dest / "trial-result.json")
        candidate = dest / f"solution_{n}.py"
        candidate_hash = None
        if candidate.exists():
            ast.parse(candidate.read_text())
            candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
        agent = result.get("agent_result") or {}
        phase = result.get("agent_execution") or {}
        elapsed = None
        if phase.get("finished_at") and phase.get("started_at"):
            elapsed = (datetime.fromisoformat(phase["finished_at"]) -
                       datetime.fromisoformat(phase["started_at"])).total_seconds()
        row = {"task": n, "raw_reward": status.get("reward"),
               "functional_score": status.get("functional_score"),
               "static_policy_score": status.get("static_policy_score"),
               "llm_audit_score": status.get("llm_audit_score"),
               "runtime_sec": status.get("runtime_sec"), "agent_wall_sec": elapsed,
               "input_tokens": agent.get("n_input_tokens"), "cache_tokens": agent.get("n_cache_tokens"),
               "output_tokens": agent.get("n_output_tokens"), "cost_usd": agent.get("cost_usd"),
               "candidate_sha256": candidate_hash, "agent_info": result.get("agent_info"),
               "exception_type": (result.get("exception_info") or {}).get("exception_type"),
               "timing": {k: result.get(k) for k in ["agent_execution", "verifier"]}}
        if row["input_tokens"] is not None and row["output_tokens"] is not None:
            row["total_tokens"] = row["input_tokens"] + row["output_tokens"]
        else:
            row["total_tokens"] = None
        row["non_cache_input_tokens"] = (row["input_tokens"] - row["cache_tokens"]
                                          if row["input_tokens"] is not None and row["cache_tokens"] is not None else None)
        assert result["config"]["agent"]["model_name"] == "gpt-6-astra"
        assert result["config"]["agent"]["kwargs"]["reasoning_effort"] == "high"
        assert result["config"]["agent"].get("resume_trajectory", False) is False
        assert not result["config"]["agent"].get("skills")
        assert not result["config"]["agent"].get("mcp_servers")
        dump(dest / "stamp-info.json", row)
        details = read_harbor_json(dest / "audit-details.json") if (dest / "audit-details.json").exists() else {}
        functional = details.get("functional", {}).get("functional_output_tail")
        if functional:
            (dest / "functional-stdout-official.txt").write_text(functional.rstrip("\n") + "\n")
        tasks.append(row)
    summary = {"label": "GPT-6 Astra high", "model": "gpt-6-astra", "effort": "high",
               "tasks": tasks, "completed_tasks": len(tasks),
               "raw_passes": sum(t["raw_reward"] == 1 for t in tasks), "final_passes": None,
               "result_basis": "Unmodified automatic Harbor verifier; no human adjudication applied"}
    for field in ["agent_wall_sec", "input_tokens", "output_tokens", "cache_tokens", "total_tokens", "cost_usd"]:
        values = [t[field] for t in tasks]
        summary[field] = sum(values) if values and all(x is not None for x in values) else None
    return summary


def main():
    models = historical()
    models["astra_high"] = astra()
    obj = {"base_commit": BASE, "models": models, "audit_model": "gpt-5.6-sol", "audit_effort": "high",
           "comparison_notes": [
               "Astra high vs Sol high changes solver model and necessarily solver CLI; it does not isolate effort.",
               "Sol high vs Sol ultra is the existing same-model effort comparison.",
               "Final adjudicated historical validity is separate from original automatic verifier reward.",
               "One trial per task and configuration; runtime differences are descriptive, not statistical estimates.",
               "Null cost means unavailable, not free. Tokens include cache reads."]}
    current = models["astra_high"]
    high, ultra = models["sol_high"], models["sol_ultra"]
    shared = [n for n in range(1, 13) if high["tasks"][n - 1]["final_valid"] and ultra["tasks"][n - 1]["final_valid"]]
    ratios = [ultra["tasks"][n - 1]["runtime_sec"] / high["tasks"][n - 1]["runtime_sec"] for n in shared]
    obj["same_model_effort_comparison"] = {
        "model": "gpt-5.6-sol", "from_effort": "high", "to_effort": "ultra",
        "final_pass_change": ultra["final_passes"] - high["final_passes"],
        "same_final_pass_set": all(h["final_valid"] == u["final_valid"] for h, u in zip(high["tasks"], ultra["tasks"])),
        "common_final_passing_tasks": shared,
        "artifact_runtime_geomean_ratio": math.exp(sum(map(math.log, ratios))/len(ratios)),
        **{f"{field}_relative_change": ultra[field] / high[field] - 1 for field in ["agent_wall_sec", "total_tokens", "cost_usd"]}}
    if current["completed_tasks"] == 12:
        comparisons = {}
        for baseline in ["sol_high", "sol_ultra"]:
            old = models[baseline]
            common = [n for n in range(1, 13) if current["tasks"][n - 1]["raw_reward"] == 1
                      and old["tasks"][n - 1]["final_valid"]]
            ratios = [current["tasks"][n - 1]["runtime_sec"] / old["tasks"][n - 1]["runtime_sec"]
                      for n in common if current["tasks"][n - 1]["runtime_sec"] > 0 and old["tasks"][n - 1]["runtime_sec"] > 0]
            comparisons[baseline] = {"common_passing_tasks": common,
                                     "runtime_geomean_ratio": math.exp(sum(map(math.log, ratios))/len(ratios)) if ratios else None}
            for field in ["agent_wall_sec", "total_tokens", "cost_usd"]:
                comparisons[baseline][f"{field}_relative_change"] = (current[field] / old[field] - 1
                                                                      if current[field] is not None and old[field] else None)
        obj["descriptive_comparisons"] = comparisons
    dump(OUT / "comparison.json", obj)
    print(json.dumps({k: {f: v.get(f) for f in ["completed_tasks", "raw_passes", "final_passes", "agent_wall_sec", "total_tokens", "cost_usd"]} for k, v in models.items()}, indent=2))


if __name__ == "__main__":
    main()
