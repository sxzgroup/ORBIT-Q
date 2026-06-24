#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

STAGE_FILES = {
    "agent_output": "agent/codex.txt",
    "agent_trajectory": "agent/trajectory.json",
    "verifier_stdout": "verifier/test-stdout.txt",
    "reward_json": "verifier/reward.json",
    "reward_txt": "verifier/reward.txt",
    "trial_result": "result.json",
    "exception": "exception.txt",
}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _file_info(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "age_sec": max(0.0, time.time() - stat.st_mtime),
    }


def _run(cmd: list[str], timeout: int = 10) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except Exception as exc:
        return False, str(exc)
    return result.returncode == 0, result.stdout


def _docker_containers(job_tokens: list[str]) -> dict[str, Any]:
    ok, output = _run(
        [
            "docker",
            "ps",
            "-a",
            "--format",
            "{{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}\t{{.Labels}}",
        ]
    )
    if not ok:
        return {"available": False, "error": output.strip(), "related": []}

    related: list[dict[str, str]] = []
    lowered_tokens = [token.lower() for token in job_tokens if token]
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) != 5:
            continue
        haystack = line.lower()
        if any(token in haystack for token in lowered_tokens):
            related.append(
                {
                    "id": fields[0],
                    "name": fields[1],
                    "status": fields[2],
                    "image": fields[3],
                    "labels": fields[4],
                }
            )
    return {"available": True, "related": related}


def _processes(job_tokens: list[str]) -> dict[str, Any]:
    ok, output = _run(["ps", "-axo", "pid,etime,command"])
    if not ok:
        return {"available": False, "error": output.strip(), "related": []}
    lowered_tokens = [token.lower() for token in job_tokens if token]
    related = [
        line
        for line in output.splitlines()
        if any(token in line.lower() for token in lowered_tokens)
        or "harbor run" in line.lower()
    ]
    return {"available": True, "related": related}


def inspect_job(job_dir: Path, stale_minutes: float) -> dict[str, Any]:
    job_dir = job_dir.resolve()
    now = time.time()
    result = _read_json(job_dir / "result.json") or {}
    lock = _read_json(job_dir / "lock.json") or {}
    trial_dirs = sorted(
        path
        for path in job_dir.iterdir()
        if path.is_dir() and (path / "config.json").exists()
    )

    files = [
        _file_info(path, job_dir)
        for path in sorted(job_dir.rglob("*"))
        if path.is_file()
    ]
    latest_age = min((item["age_sec"] for item in files), default=None)

    trials: list[dict[str, Any]] = []
    tokens = [job_dir.name]
    for trial_dir in trial_dirs:
        tokens.append(trial_dir.name)
        cfg = _read_json(trial_dir / "config.json") or {}
        task_path = cfg.get("task", {}).get("path")
        if task_path:
            tokens.append(Path(task_path).name)
        present = {key: (trial_dir / rel).exists() for key, rel in STAGE_FILES.items()}
        stage_files = {
            key: _file_info(trial_dir / rel, job_dir)
            for key, rel in STAGE_FILES.items()
            if (trial_dir / rel).exists()
        }
        trials.append(
            {
                "name": trial_dir.name,
                "present": present,
                "stage_files": stage_files,
            }
        )

    docker = _docker_containers(tokens)
    processes = _processes(tokens)

    stats = result.get("stats", {})
    finished = result.get("finished_at") is not None
    running_trials = int(stats.get("n_running_trials") or 0)
    completed_or_errored = int(stats.get("n_completed_trials") or 0) + int(
        stats.get("n_errored_trials") or 0
    )
    stale_threshold_sec = stale_minutes * 60.0
    has_stage_artifacts = any(any(trial["present"].values()) for trial in trials)
    docker_running = any(
        "Up " in item.get("status", "") for item in docker.get("related", [])
    )
    proc_running = (
        bool(processes.get("related")) if processes.get("available") else False
    )
    no_recent_files = latest_age is None or latest_age > stale_threshold_sec

    if finished or (running_trials == 0 and completed_or_errored > 0):
        status = "completed"
    elif docker_running or proc_running:
        status = "active"
    elif running_trials > 0 and no_recent_files and not has_stage_artifacts:
        status = "suspect_stale_orchestration"
    elif running_trials > 0 and no_recent_files:
        status = "suspect_stale_stage"
    elif running_trials > 0:
        status = "running_or_recently_active"
    else:
        status = "unknown"

    confidence = "high"
    if not docker.get("available") or not processes.get("available"):
        confidence = "medium"
    if status.startswith("suspect") and (docker_running or proc_running):
        confidence = "low"

    return {
        "job_dir": str(job_dir),
        "status": status,
        "confidence": confidence,
        "now": now,
        "latest_file_age_sec": latest_age,
        "stale_threshold_sec": stale_threshold_sec,
        "result_summary": {
            "finished_at": result.get("finished_at"),
            "stats": stats,
        },
        "lock_invocation": lock.get("invocation"),
        "trials": trials,
        "docker": docker,
        "processes": processes,
        "file_count": len(files),
        "largest_recent_files": sorted(
            files, key=lambda item: item["mtime"], reverse=True
        )[:20],
    }


def _print_text(report: dict[str, Any]) -> None:
    print(f"status: {report['status']} ({report['confidence']} confidence)")
    print(f"job: {report['job_dir']}")
    print(f"latest_file_age_sec: {report['latest_file_age_sec']}")
    stats = report["result_summary"].get("stats") or {}
    print(f"stats: {stats}")
    print(f"docker_available: {report['docker']['available']}")
    if not report["docker"]["available"]:
        print(f"docker_error: {report['docker'].get('error')}")
    print(f"related_containers: {len(report['docker'].get('related', []))}")
    for item in report["docker"].get("related", []):
        print(
            f"  container: {item['id']} {item['name']} {item['status']} {item['image']}"
        )
    print(f"process_table_available: {report['processes']['available']}")
    if not report["processes"]["available"]:
        print(f"process_error: {report['processes'].get('error')}")
    for line in report["processes"].get("related", [])[:20]:
        print(f"  process: {line}")
    for trial in report["trials"]:
        present = (
            ", ".join(key for key, value in trial["present"].items() if value) or "none"
        )
        print(f"trial {trial['name']} stage_files: {present}")
    print("recent files:")
    for item in report["largest_recent_files"][:10]:
        print(f"  {item['age_sec']:.1f}s {item['size']} {item['path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a Harbor job directory.")
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--stale-minutes", type=float, default=5.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = inspect_job(args.job_dir, args.stale_minutes)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_text(report)


if __name__ == "__main__":
    main()
