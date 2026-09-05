#!/usr/bin/env python3
"""Host-only provenance check; use verify_archive.py for offline PR review."""
import hashlib
import json
import re
import subprocess
from run_benchmark import ROOT, OUT, RUN, JOBS, BASE, command, hashes, dump

if not RUN.is_dir() or not JOBS.is_dir():
    raise SystemExit("Host-only check requires original task copies and job logs. "
                     "For offline archive review, run verify_archive.py instead.")

protocol = json.loads((OUT / "protocol.json").read_text())
rows = []
needles = ["gpt56sol-high", "gpt56sol-ultra", "gpt-5.6-sol-high-orbit-q",
           "github.com/sxzgroup/orbit-q/pull/5", "github.com/sxzgroup/orbit-q/pull/6",
           "/users/qqy/desktop/2026project/orbit-q/.scratch"]
subprocess.run(["git", "diff", "--exit-code", BASE, "--", "tasks", "adapters", "prompts"], cwd=ROOT, check=True)
for n in range(1, 13):
    folder = OUT / f"challenge-{n:02d}"
    if not (folder / "status.json").exists():
        continue
    trial = next((JOBS / command(n)[1]).glob(f"challenge-{n:02d}__*"))
    cfg = json.loads((folder / "trial-result.json").read_text())["config"]
    assert cfg["agent"]["model_name"] == "gpt-6-astra"
    assert cfg["agent"]["kwargs"]["reasoning_effort"] == "high"
    assert cfg["agent"]["resume_trajectory"] is False
    assert cfg["agent"].get("load_trajectory") is None
    assert cfg["agent"]["skills"] == [] and cfg["agent"]["mcp_servers"] == []
    assert cfg["verifier"]["kwargs"]["audit_model"] == "gpt-5.6-sol"
    assert cfg["verifier"]["env"]["PATH"] == "/usr/local/bin:/usr/bin:/bin"
    assert cfg["environment"].get("mounts") is None
    assert hashes(RUN / f"challenge-{n:02d}") == protocol["task_hashes"][f"challenge-{n:02d}"]
    sessions, contexts = [], []
    for path in sorted((trial / "agent/sessions").rglob("*.jsonl")):
        for line in path.read_text(errors="replace").splitlines():
            try:
                event = json.loads(re.sub(r'(:\s*)\[REDACTED\](?=\s*[,}\]])', r'\1null', line))
            except ValueError:
                continue
            payload = event.get("payload", {})
            if event.get("type") == "session_meta":
                sessions.append({k: payload.get(k) for k in ["id", "cli_version", "cwd", "history_mode"]})
            if event.get("type") == "turn_context":
                contexts.append({k: payload.get(k) for k in ["model", "effort", "cwd"]})
    assert sessions and contexts, n
    assert all(c["model"] == "gpt-6-astra" and c["effort"] == "high" for c in contexts), contexts
    assert all(s["cli_version"] == "0.153.3" and s["cwd"] == "/root" for s in sessions), sessions
    text = (trial / "agent/codex.txt").read_text(errors="replace").lower()
    matches = {p: text.count(p) for p in needles}
    mounts = []
    for path in sorted((OUT / "container-evidence").glob(f"challenge-{n:02d}__*.json")):
        snapshot = json.loads(path.read_text())
        assert snapshot["image"] == protocol["docker_image_id"]
        assert snapshot["nano_cpus"] == 6000000000 and snapshot["memory_bytes"] == 10737418240
        assert all(m["Source"].startswith(str(trial) + "/") for m in snapshot["mounts"])
        assert all(m["Destination"] in ["/logs/agent", "/logs/verifier", "/logs/artifacts"] for m in snapshot["mounts"])
        mounts.append(path.name)
    assert mounts, n
    candidate = folder / f"solution_{n}.py"
    source = trial / "artifacts/root" / f"solution_{n}.py"
    assert candidate.read_bytes() == source.read_bytes()
    for name, upstream in [("audit-details", trial / "verifier/audit-details.json"),
                           ("trial-result", trial / "result.json")]:
        original = folder / f"{name}.raw.txt"
        if not original.exists():
            original = folder / f"{name}.json"
        assert original.read_bytes() == upstream.read_bytes(), (n, name)
    rows.append({"task": n, "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                 "task_copy_unchanged": True, "raw_verifier_provenance_unchanged": True,
                 "fresh_session_ids": [s["id"] for s in sessions],
                 "session_metadata": sessions, "observed_contexts": contexts,
                 "old_result_path_matches": matches, "container_evidence": mounts,
                 "resume_trajectory": False, "skills": [], "mcp_servers": [], "extra_mounts": None})
ids = [session for row in rows for session in row["fresh_session_ids"]]
if len(rows) != 12:
    raise ValueError("Incomplete campaign: expected all twelve tasks")
assert len(ids) == len(set(ids)), "A session was reused across tasks"
obj = {"checked_tasks": len(rows), "all_unique_sessions": True,
       "scope": "Host mounts, task hashes, explicit configuration, recorded CLI/model/effort, and exact historical-result path strings.",
       "conclusion": "No evidence of exposure to previous results in the recorded inputs and mounts" if not any(any(row["old_result_path_matches"].values()) for row in rows) else "Inspect transcript path matches before concluding",
       "limitations": "This establishes execution isolation, not absence of prior model training knowledge. Keyword checks are supplementary to mount/config evidence.",
       "tasks": rows}
dump(OUT / "contamination-audit.json", obj)
print(json.dumps({"checked_tasks": len(rows), "conclusion": obj["conclusion"]}))
