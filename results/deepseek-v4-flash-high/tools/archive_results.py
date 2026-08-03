"""Archive the selected valid DeepSeek V4 Flash/high campaign outcomes."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "deepseek-v4-flash-high"
CAMPAIGN = ROOT / "jobs" / "deepseek-v4-flash-high-solaudit-20260802-valid"
ATTEMPTS = {task: "r1" for task in range(1, 13)}


def load(path: Path) -> dict:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(text.replace("[REDACTED]", "null"))


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seconds_between(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    begin = datetime.fromisoformat(start.replace("Z", "+00:00"))
    finish = datetime.fromisoformat(end.replace("Z", "+00:00"))
    return (finish - begin).total_seconds()


def selected_trial(task: int) -> tuple[Path, Path]:
    attempt = ATTEMPTS[task]
    job = CAMPAIGN / "jobs" / (
        f"challenge-{task:02d}-tensorcircuit-deepseek-v4-flash-high-20260802-{attempt}"
    )
    trials = sorted(job.glob(f"challenge-{task:02d}__*"))
    if len(trials) != 1:
        raise RuntimeError(f"Expected one trial for task {task:02d}, found {trials}")
    return job, trials[0]


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> None:
    task_rows = []
    for task in range(1, 13):
        job, trial = selected_trial(task)
        challenge_out = OUT / f"challenge-{task:02d}"
        challenge_out.mkdir(parents=True, exist_ok=True)

        solution = trial / "artifacts" / "root" / f"solution_{task}.py"
        copies = {
            solution: challenge_out / f"solution_{task}.py",
            trial / "verifier" / "functional-stdout.txt": challenge_out / "functional-stdout-official.txt",
            trial / "verifier" / "audit-details.json": challenge_out / "audit-details.json",
            trial / "verifier" / "reward.json": challenge_out / "reward.json",
            trial / "result.json": challenge_out / "job-result.json",
            trial / "config.json": challenge_out / "config.json",
            trial / "lock.json": challenge_out / "lock.json",
            trial / "agent" / "codex.txt": challenge_out / "agent-codex.log",
            trial / "trial.log": challenge_out / "trial.log",
            trial / "artifacts" / "manifest.json": challenge_out / "artifact-manifest.json",
        }
        for source, target in copies.items():
            copy_if_exists(source, target)

        result = load(trial / "result.json")
        reward = load(trial / "verifier" / "reward.json")
        config = result["config"]
        agent_result = result.get("agent_result") or {}
        agent_interval = result.get("agent_execution") or {}
        stamp = {
            "challenge": task,
            "selected_attempt": ATTEMPTS[task],
            "job_name": job.name,
            "trial_name": trial.name,
            "solver_model": config["agent"]["model_name"],
            "solver_reasoning_effort": config["agent"]["kwargs"]["reasoning_effort"],
            "audit_model": config["verifier"]["kwargs"]["audit_model"],
            "audit_reasoning_effort": "high",
            "framework": config["environment"]["kwargs"]["framework"],
            "docker_image": config["environment"]["kwargs"]["docker_image"],
            "started_at": result.get("started_at"),
            "finished_at": result.get("finished_at"),
            "agent_wall_time_sec": seconds_between(
                agent_interval.get("started_at"), agent_interval.get("finished_at")
            ),
            "n_input_tokens": agent_result.get("n_input_tokens"),
            "n_cache_tokens": agent_result.get("n_cache_tokens"),
            "n_output_tokens": agent_result.get("n_output_tokens"),
            "cost_usd": agent_result.get("cost_usd"),
            "solution_sha256": sha256(solution),
            "reward": reward["reward"],
            "functional_score": reward["functional_score"],
            "runtime_score": reward["runtime_score"],
            "static_policy_score": reward["static_policy_score"],
            "llm_audit_score": reward["llm_audit_score"],
            "runtime_sec": reward["runtime_sec"],
            "exception_type": (result.get("exception_info") or {}).get("exception_type"),
        }
        dump(challenge_out / "stamp-info.json", stamp)
        task_rows.append(stamp)

    manifest = load(CAMPAIGN / "task-copy-manifest.json")
    copy_if_exists(CAMPAIGN / "task-copy-manifest.json", OUT / "task-copy-manifest.json")
    measured = [row["runtime_sec"] for row in task_rows if row["runtime_sec"] >= 0]
    summary = {
        "schema_version": 1,
        "protocol": {
            "base_commit": manifest["base_commit"],
            "solver_model": "deepseek-v4-flash",
            "solver_reasoning_effort": "high",
            "audit_model": "gpt-5.6-sol",
            "audit_reasoning_effort": "high",
            "framework": "tensorcircuit",
            "docker_image": "challenge-benchmark-quantum-tensorcircuit:py311",
            "execution_order": "sequential",
            "trials_per_valid_outcome": 1,
            "resource_adapter": manifest["resource_adapter"],
            "aggregate_task_copy_sha256": manifest["aggregate_task_copy_sha256"],
        },
        "counts": {
            "valid_outcomes": 12,
            "passes": sum(row["reward"] == 1 for row in task_rows),
            "functional_passes": sum(row["functional_score"] == 1 for row in task_rows),
            "static_policy_passes": sum(row["static_policy_score"] == 1 for row in task_rows),
            "audit_passes": sum(row["llm_audit_score"] == 1 for row in task_rows),
        },
        "runtime": {
            "measured_tasks": len(measured),
            "total_sec": round(sum(measured), 2),
            "mean_sec": round(sum(measured) / len(measured), 2),
        },
        "agent_usage": {
            "wall_time_sec": round(sum(row["agent_wall_time_sec"] or 0 for row in task_rows), 3),
            "input_tokens": sum(row["n_input_tokens"] or 0 for row in task_rows),
            "cache_tokens": sum(row["n_cache_tokens"] or 0 for row in task_rows),
            "output_tokens": sum(row["n_output_tokens"] or 0 for row in task_rows),
            "cost_usd": round(sum(row["cost_usd"] or 0 for row in task_rows), 6),
        },
        "tasks": task_rows,
        "excluded_attempts": {},
        "notes": {
            "agent_timeouts": "Challenges 01, 02, 08, 09, and 11 reached the 1,800-second Agent limit without a submitted candidate. Challenge 05 reached the limit after writing a candidate; Harbor verified it normally and awarded reward 1.",
            "audit_failures": "Challenges 06 and 07 passed functional and static checks but failed the independent Sol/high source audit.",
            "runtime": "runtime_score is retained for reporting and does not multiply the pass reward.",
        },
    }
    dump(OUT / "summary.json", summary)


if __name__ == "__main__":
    main()
