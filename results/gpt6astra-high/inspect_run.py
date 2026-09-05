#!/usr/bin/env python3
"""Read-only monitoring; record actual mounts/resources without environment secrets."""
import json
import subprocess
import time
from pathlib import Path
from run_benchmark import OUT, RUN, JOBS, now, dump

active = json.loads((RUN / "active.json").read_text())
finished = [json.loads(p.read_text()) for p in sorted(OUT.glob("challenge-*/status.json"))]
summary = {"at": now(), "active": active, "completed": len(finished),
           "passed": sum(t.get("reward") == 1 for t in finished),
           "scores": {f"{t['challenge']:02d}": t.get("reward") for t in finished}}
if "job" in active:
    directory = JOBS / active["job"]
    for path in directory.glob("*/agent/codex.txt"):
        events = []
        for line in path.read_text(errors="replace").splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
        summary["agent_events"] = len(events)
        summary["log_age_sec"] = round(time.time() - path.stat().st_mtime, 1)
        summary["errors"] = [e for e in events if e.get("type") in ["error", "turn.failed"]][-2:]
        summary["turn_completed"] = any(e.get("type") == "turn.completed" for e in events)
    for path in directory.glob("*/verifier/reward.json"):
        summary["pending_archive_reward"] = json.loads(path.read_text())
containers = subprocess.check_output(["docker", "ps", "--format", "{{.ID}} {{.Names}}"], text=True)
summary["containers"] = []
for line in containers.splitlines():
    cid, name = line.split(maxsplit=1)
    if not name.startswith("challenge-"):
        continue
    details = json.loads(subprocess.check_output(["docker", "inspect", cid], text=True))[0]
    # Restrict evidence to this run's logs. Never expose Config.Env or auth.
    if not any(str(JOBS) in m.get("Source", "") for m in details.get("Mounts", [])):
        continue
    snapshot = {"observed_at": now(), "id": details["Id"], "name": name,
                "image": details["Image"], "mounts": details["Mounts"],
                "nano_cpus": details["HostConfig"]["NanoCpus"],
                "memory_bytes": details["HostConfig"]["Memory"],
                "started_at": details["State"]["StartedAt"]}
    dump(OUT / "container-evidence" / f"{name}.json", snapshot)
    summary["containers"].append({"name": name, "cpus": snapshot["nano_cpus"] / 1e9,
                                  "memory_gib": snapshot["memory_bytes"] / 2**30})
print(json.dumps(summary, indent=2))
