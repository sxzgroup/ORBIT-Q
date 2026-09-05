#!/usr/bin/env python3
"""Read-only archive verification: Python 3.10+, standard library, no services.

Checks stored evidence and accounting, not algorithm correctness or unarchived
solver logs. Does not import candidates, execute the benchmark, or rewrite data.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import struct
import sys
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET
import zlib

ROOT = Path(__file__).resolve().parent
TASKS = list(range(1, 13))
BASE = "0201238ec2983907e2891f5319f5fff2d00844d5"
TASK_HASH_DIGEST = "d97ed668b5ea550ea523d6a5bdc8a1861ad7671c29aafb45729e728ae3930767"
PUBLIC_DIGEST = "4950fd4a7c81faa83c2486b910935cee2630b8717920ea68922b0fac99f79217"
PARENT_IMAGE = "sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833"
IMAGE = "sha256:ce64a50ab90eaee930d981b7397903bc4baca54f32b8c88f5fa6ad2c40b9f5d7"
FIGURES = ["astra-high-vs-sol-outcomes", "astra-high-agent-resource-use"]
RESOURCE_FIELDS = ["agent_wall_sec", "input_tokens", "cache_tokens", "output_tokens", "total_tokens", "cost_usd"]


class VerificationError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise VerificationError(message)


def close(actual, expected, label):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8), label)


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(text):
    def reject_constant(value):
        raise VerificationError(f"Invalid JSON constant: {value}")
    return json.loads(text, object_pairs_hook=unique_keys, parse_constant=reject_constant)


def load(path):
    return parse_json(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    files = {}
    for path in root.rglob("*"):
        require(not path.is_symlink(), f"Unexpected symlink: {path.relative_to(root)}")
        if path.is_file() and "__pycache__" not in path.parts:
            relative = path.relative_to(root).as_posix()
            if relative != "archive-manifest.json":
                files[relative] = sha(path)
    return dict(sorted(files.items()))


def verify_manifest(root):
    manifest = load(root / "archive-manifest.json")
    require(manifest["schema_version"] == 1, "Unsupported archive manifest")
    actual = inventory(root)
    expected = manifest["sha256"]
    require(actual.keys() == expected.keys(), "Archive file inventory mismatch (missing or extra file)")
    for name, digest in actual.items():
        require(digest == expected[name], f"Archive checksum mismatch: {name}")
    return len(actual)


def verify_normalization(folder, name):
    obj = load(folder / f"{name}.json")
    metadata = obj.pop("_archive_normalization")
    require(metadata["original_file"] == f"{name}.raw.txt", f"{name}: raw filename")
    raw = (folder / metadata["original_file"]).read_text()
    normalized, count = re.subn(r'(:\s*)\[REDACTED\](?=\s*[,}\]])', r'\1null', raw)
    require(count == metadata["redacted_values_represented_as_null"] and count > 0,
            f"{folder.name}/{name}: normalization count")
    require(obj == parse_json(normalized), f"{folder.name}/{name}: raw/normalized mismatch")
    return obj


def verify_protocol(root):
    protocol = load(root / "protocol.json")
    fixed = {"base_commit": BASE, "solver_model": "gpt-6-astra", "solver_effort": "high",
             "audit_model": "gpt-5.6-sol", "audit_effort": "high", "solver_cli": "0.153.3",
             "audit_cli": "0.145.0", "harbor_version": "0.20.0", "parent_image_id": PARENT_IMAGE,
             "docker_image_id": IMAGE, "cpus": 6, "memory_mb": 10240, "storage_mb": 16384,
             "trials_per_challenge": 1, "harbor_retries": 0, "concurrency": 1,
             "execution_order": TASKS, "resume": False, "skills": [], "mcp_servers": [],
             "extra_host_mounts": []}
    for key, value in fixed.items():
        require(protocol[key] == value, f"Protocol mismatch: {key}")
    digest = hashlib.sha256(json.dumps(protocol["task_hashes"], sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    require(digest == TASK_HASH_DIGEST, "Pinned task-hash manifest changed")
    require((root / "audit-high.config.toml").read_text().strip() == 'model_reasoning_effort = "high"',
            "Audit effort config mismatch")
    return protocol


def task_rows(rows, label):
    require([r["task"] for r in rows] == TASKS, f"{label}: expected exactly tasks 01–12 in order")
    return rows


def verify_task(root, row, evidence, protocol):
    n = row["task"]
    folder = root / f"challenge-{n:02d}"
    reward, stamp, status, job = [load(folder / name) for name in
                                 ["reward.json", "stamp-info.json", "status.json", "job-result.json"]]
    trial = verify_normalization(folder, "trial-result")
    audit = verify_normalization(folder, "audit-details")
    require(reward["problem_id"] == n and audit["problem_id"] == n, f"Task {n}: problem ID")
    require(reward == trial["verifier_result"]["rewards"], f"Task {n}: verifier reward")
    close(reward["reward"], math.prod(reward[k] for k in
          ["functional_score", "static_policy_score", "llm_audit_score"]), f"Task {n}: reward product")
    for field in ["reward", "functional_score", "static_policy_score", "llm_audit_score"]:
        close(reward[field], 1, f"Task {n}: {field}")
        key = "raw_reward" if field == "reward" else field
        close(row[key], reward[field], f"Task {n}: comparison {field}")
        close(stamp[key], reward[field], f"Task {n}: stamp {field}")
        close(status[field], reward[field], f"Task {n}: status {field}")
    require(status["harbor_exit_code"] == 0 and status["status"] == "finished", f"Task {n}: status")
    require(trial["exception_info"] is None and row["exception_type"] is None, f"Task {n}: exception")
    require(job["n_total_trials"] == job["stats"]["n_completed_trials"] == 1
            and job["stats"]["n_errored_trials"] == job["stats"]["n_retries"] == 0,
            f"Task {n}: trial/retry counts")
    require(trial["config"]["job_id"] == job["id"], f"Task {n}: job identity")
    cfg = trial["config"]
    agent, verifier, environment = cfg["agent"], cfg["verifier"], cfg["environment"]
    require(agent["model_name"] == "gpt-6-astra" and agent["kwargs"]["reasoning_effort"] == "high",
            f"Task {n}: solver model/effort")
    require(agent["resume_trajectory"] is False and agent["load_trajectory"] is None
            and agent["skills"] == agent["mcp_servers"] == [], f"Task {n}: unexpected solver input")
    require(verifier["kwargs"]["audit_model"] == "gpt-5.6-sol"
            and Path(verifier["kwargs"]["profile_config_path"]).name == "audit-high.config.toml"
            and verifier["env"]["PATH"] == "/usr/local/bin:/usr/bin:/bin"
            and verifier["env"]["REQUIRED_QUANTUM_FRAMEWORK"] == "tensorcircuit", f"Task {n}: verifier config")
    require(environment["kwargs"]["docker_image"] == protocol["docker_image"]
            and environment["kwargs"]["framework"] == "tensorcircuit"
            and environment["mounts"] is None, f"Task {n}: container config")
    require(trial["agent_info"]["version"] == "0.153.3"
            and trial["agent_info"]["model_info"]["name"] == "gpt-6-astra", f"Task {n}: actual solver metadata")
    candidate = folder / f"solution_{n}.py"
    ast.parse(candidate.read_text())  # Never import or execute candidate code.
    require(sha(candidate) == row["candidate_sha256"] == stamp["candidate_sha256"]
            == evidence["candidate_sha256"], f"Task {n}: candidate hash")
    require(audit["static"]["parse_ok"] == 1 and audit["static"]["line_count"] <= 200,
            f"Task {n}: static record")
    for group, field in [("static", "static_policy_score"), ("functional", "functional_score"), ("audit", "llm_audit_score")]:
        close(audit[group][field], reward[field], f"Task {n}: audit {field}")
    output = (folder / "functional-stdout-official.txt").read_text()
    require(output.strip() == audit["functional"]["functional_output_tail"].strip(), f"Task {n}: functional output")
    match = re.search(r"End-to-end solution time: ([0-9.]+)s", output)
    require(match is not None and "Overall: PASS" in output, f"Task {n}: functional runtime/pass line")
    for recorded in [row["runtime_sec"], stamp["runtime_sec"], audit["functional"]["runtime_sec"], float(match[1])]:
        close(recorded, reward["runtime_sec"], f"Task {n}: runtime")
    stats = trial["agent_result"]
    for field in ["input_tokens", "cache_tokens", "output_tokens"]:
        close(row[field], stats["n_" + field], f"Task {n}: {field}")
    close(row["cost_usd"], stats["cost_usd"], f"Task {n}: solver cost")
    phase = trial["agent_execution"]
    seconds = (datetime.fromisoformat(phase["finished_at"].replace("Z", "+00:00")) -
               datetime.fromisoformat(phase["started_at"].replace("Z", "+00:00"))).total_seconds()
    close(row["agent_wall_sec"], seconds, f"Task {n}: solver wall time")
    for field in RESOURCE_FIELDS:
        close(stamp[field], row[field], f"Task {n}: stamp {field}")
    close(row["non_cache_input_tokens"], row["input_tokens"] - row["cache_tokens"], f"Task {n}: non-cache tokens")
    require(evidence["fresh_session_ids"] == [s["id"] for s in evidence["session_metadata"]]
            and len(evidence["fresh_session_ids"]) == 1, f"Task {n}: session identity")
    require(all(s["cli_version"] == "0.153.3" and s["cwd"] == "/root" for s in evidence["session_metadata"])
            and evidence["observed_contexts"] and all(c["model"] == "gpt-6-astra" and c["effort"] == "high"
            for c in evidence["observed_contexts"]), f"Task {n}: session evidence")
    require(evidence["old_result_path_matches"] and not any(evidence["old_result_path_matches"].values()),
            f"Task {n}: historical path matches")
    require(evidence["container_evidence"], f"Task {n}: missing container evidence")
    trial_path = unquote(urlparse(trial["trial_uri"]).path)
    for filename in evidence["container_evidence"]:
        require(Path(filename).name == filename, f"Task {n}: invalid evidence filename")
        snapshot = load(root / "container-evidence" / filename)
        require(snapshot["image"] == IMAGE and snapshot["nano_cpus"] == 6_000_000_000
                and snapshot["memory_bytes"] == 10_737_418_240, f"Task {n}: actual container resources")
        require(snapshot["mounts"] and all(m["Source"].startswith(trial_path + "/")
                and m["Destination"] in ["/logs/agent", "/logs/verifier", "/logs/artifacts"]
                for m in snapshot["mounts"]), f"Task {n}: container mounts")
    return evidence["fresh_session_ids"][0]


def geomean(values):
    return math.exp(sum(math.log(x) for x in values) / len(values))


def verify_comparisons(root, data):
    models = data["models"]
    require(set(models) == {"sol_high", "sol_ultra", "astra_high"}, "Unexpected comparison models")
    for name, model in models.items():
        rows = task_rows(model["tasks"], name)
        for r in rows:
            require(0 <= r["cache_tokens"] <= r["input_tokens"], f"{name}: cache accounting")
            close(r["total_tokens"], r["input_tokens"] + r["output_tokens"], f"{name}: token accounting")
        for field in RESOURCE_FIELDS:
            close(model[field], sum(r[field] for r in rows), f"{name}: total {field}")
        require(model["raw_passes"] == sum(r["raw_reward"] == 1 for r in rows), f"{name}: automatic passes")
        if name != "astra_high":
            require(model["final_passes"] == sum(r["final_valid"] for r in rows) == 10, f"{name}: final passes")
    high, ultra, astra = (models[k] for k in ["sol_high", "sol_ultra", "astra_high"])
    require(astra["raw_passes"] == astra["completed_tasks"] == 12 and astra["final_passes"] is None,
            "Astra outcome/review basis")
    common = [h["task"] for h, u in zip(high["tasks"], ultra["tasks"]) if h["final_valid"] and u["final_valid"]]
    effort = data["same_model_effort_comparison"]
    require(effort["common_final_passing_tasks"] == common and effort["same_final_pass_set"] is True
            and effort["final_pass_change"] == 0, "Effort outcome basis")
    close(effort["artifact_runtime_geomean_ratio"], geomean([ultra["tasks"][n-1]["runtime_sec"] /
          high["tasks"][n-1]["runtime_sec"] for n in common]), "Effort runtime ratio")
    for field in ["agent_wall_sec", "total_tokens", "cost_usd"]:
        close(effort[field + "_relative_change"], ultra[field]/high[field]-1, f"Effort {field}")
    for key in ["sol_high", "sol_ultra"]:
        baseline = models[key]
        ratio = data["descriptive_comparisons"][key]
        require(ratio["common_passing_tasks"] == common, f"{key}: runtime task set")
        close(ratio["runtime_geomean_ratio"], geomean([astra["tasks"][n-1]["runtime_sec"] /
              baseline["tasks"][n-1]["runtime_sec"] for n in common]), f"{key}: runtime ratio")
        for field in ["agent_wall_sec", "total_tokens", "cost_usd"]:
            close(ratio[field + "_relative_change"], astra[field]/baseline[field]-1, f"{key}: resource ratio")
    public_path = root / "public-model-comparison.json"
    require(sha(public_path) == PUBLIC_DIGEST, "Pinned PR #28 comparison data changed")
    public = load(public_path)
    require(len(public["runs"]) == 8 and len({r["label"] for r in public["runs"]}) == 8, "Historical model count")
    for run in public["runs"]:
        require(run["pass_count"] == len(set(run["passes"])) and set(run["passes"]) <= set(TASKS), "Historical pass set")
        require(run["total_cost_usd"] > 0 and run["agent_wall_sec"] > 0, "Historical resource totals")
    for name, run in zip(["sol_high", "sol_ultra"], public["runs"][:2]):
        model = models[name]
        require(run["passes"] == [r["task"] for r in model["tasks"] if r["final_valid"]], "Cross-file Sol pass set")
        close(run["total_cost_usd"], model["cost_usd"], "Cross-file Sol cost")
        close(run["agent_wall_sec"], model["agent_wall_sec"], "Cross-file Sol time")


def verify_figures(root):
    expected = {f"{stem}.{ext}" for stem in FIGURES for ext in ["png", "pdf", "svg"]}
    require({p.name for p in (root / "figs").iterdir()} == expected, "Expected exactly two figures in three formats")
    for stem in FIGURES:
        png = (root / "figs" / f"{stem}.png").read_bytes()
        require(png[:8] == b"\x89PNG\r\n\x1a\n", f"{stem}: PNG signature")
        offset, kinds = 8, []
        while offset < len(png):
            require(offset + 12 <= len(png), f"{stem}: truncated PNG chunk")
            size = struct.unpack(">I", png[offset:offset+4])[0]
            end = offset + 12 + size
            require(end <= len(png), f"{stem}: PNG chunk length")
            kind = png[offset+4:offset+8]
            content = png[offset+8:end-4]
            require(zlib.crc32(kind + content) == struct.unpack(">I", png[end-4:end])[0], f"{stem}: PNG CRC")
            if kind == b"IHDR":
                width, height = struct.unpack(">II", content[:8])
                require(width >= 2500 and height >= 800, f"{stem}: PNG dimensions")
            kinds.append(kind)
            offset = end
        require(kinds[0] == b"IHDR" and kinds[-1] == b"IEND" and b"IDAT" in kinds, f"{stem}: PNG structure")
        pdf = (root / "figs" / f"{stem}.pdf").read_bytes()
        require(pdf.startswith(b"%PDF-") and pdf.rstrip().endswith(b"%%EOF"), f"{stem}: PDF structure")
        svg = ET.fromstring((root / "figs" / f"{stem}.svg").read_text())
        require(svg.tag.endswith("}svg"), f"{stem}: SVG structure")
        text = " ".join(svg.itertext())
        require(all(run["label"] in text for run in load(root / "public-model-comparison.json")["runs"]
                    if not run["label"].startswith("DeepSeek")) and "Astra high" in text,
                f"{stem}: model labels")


def verify_square_efficiency(root):
    folder = root / "resource-efficiency"
    stem = "astra-high-resource-efficiency-square"
    require({p.name for p in folder.iterdir()} == {f"{stem}.{ext}" for ext in ("png", "pdf", "svg")},
            "Square efficiency figure formats")
    png = (folder / f"{stem}.png").read_bytes()
    require(png.startswith(b"\x89PNG\r\n\x1a\n"), "Square efficiency PNG signature")
    width, height = struct.unpack(">II", png[16:24])
    require(width == height == 1536, "Square efficiency PNG dimensions")
    pdf = (folder / f"{stem}.pdf").read_bytes()
    require(pdf.startswith(b"%PDF-") and pdf.rstrip().endswith(b"%%EOF"), "Square efficiency PDF")
    svg = ET.fromstring((folder / f"{stem}.svg").read_text())
    viewbox = list(map(float, svg.attrib["viewBox"].split()))
    require(viewbox[2] == viewbox[3], "Square efficiency SVG dimensions")
    text = " ".join(" ".join(svg.itertext()).split())
    labels = [r["label"] for r in load(root / "public-model-comparison.json")["runs"]] + ["Astra high"]
    require(all(label in text for label in labels) and "total solver cost" in text,
            "Square efficiency model labels and area legend")


def verify(root=ROOT):
    root = Path(root).resolve()
    count = verify_manifest(root)
    for path in root.rglob("*.json"):
        load(path)
    for path in root.rglob("*.py"):
        ast.parse(path.read_text())
    require(sorted(p.name for p in root.glob("challenge-*")) == [f"challenge-{n:02d}" for n in TASKS],
            "Expected exactly twelve challenge directories")
    protocol = verify_protocol(root)
    data, isolation = load(root / "comparison.json"), load(root / "contamination-audit.json")
    require(data["base_commit"] == BASE and data["audit_model"] == "gpt-5.6-sol"
            and data["audit_effort"] == "high", "Comparison protocol")
    require(isolation["checked_tasks"] == 12 and isolation["all_unique_sessions"] is True, "Isolation task count")
    rows = task_rows(data["models"]["astra_high"]["tasks"], "Astra")
    evidence = task_rows(isolation["tasks"], "Isolation evidence")
    sessions = [verify_task(root, row, item, protocol) for row, item in zip(rows, evidence)]
    require(len(set(sessions)) == 12, "Session reused across tasks")
    verify_comparisons(root, data)
    verify_figures(root)
    verify_square_efficiency(root)
    try:
        runtime = runpy.run_path(str(root / "runtime/verify_runtime.py"))["verify"](root / "runtime")
    except ValueError as exc:
        raise VerificationError(f"Runtime supplement: {exc}") from exc
    for path in root.glob("*.md"):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if not target.startswith("https://"):
                require((path.parent / target).exists(), f"Broken link in {path.name}: {target}")
    return {"files_checked": count, "tasks": 12, "automatic_passes": 12, "comparison_models": 9,
            "figure_files": 6, "square_efficiency_files": 3, "runtime_figure_files": 4,
            "runtime_comparisons": runtime["task_count"],
            "estimated_expert_references": runtime["estimated_reference_count"],
            "mode": "offline, read-only; stored evidence only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        result = verify(args.archive_dir)
    except (OSError, ValueError, KeyError, TypeError, SyntaxError, ET.ParseError, struct.error) as exc:
        print(f"Archive verification FAILED: {exc}", file=sys.stderr)
        return 1
    print("Archive verification passed: " + json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
