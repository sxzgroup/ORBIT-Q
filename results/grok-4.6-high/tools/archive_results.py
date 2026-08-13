"""Archive the selected valid Grok 4.6/high campaign outcomes."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "grok-4.6-high"
HOST_REPO = Path("/Users/qqy/Desktop/2026Project/ORBIT-Q")
CAMPAIGN = HOST_REPO / "jobs" / "grok-4.6-high-solaudit-20260813-valid"
ATTEMPTS = {task: "r1" for task in range(1, 13)}
TC_VERSION = "1.8.0.dev20260726"
IMAGE_ID = "sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833"


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
        f"challenge-{task:02d}-tensorcircuit-grok-4.6-high-20260813-{attempt}"
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
        if challenge_out.exists():
            shutil.rmtree(challenge_out)
        challenge_out.mkdir(parents=True, exist_ok=True)

        solution = trial / "artifacts" / "root" / f"solution_{task}.py"
        copies = {
            solution: challenge_out / f"solution_{task}.py",
            trial / "verifier" / "functional-stdout.txt": challenge_out / "functional-stdout-official.txt",
            trial / "verifier" / "test-stdout.txt": challenge_out / "verifier-test-stdout.txt",
            trial / "verifier" / "audit-details.json": challenge_out / "audit-details.json",
            trial / "verifier" / "reward.json": challenge_out / "reward.json",
            trial / "result.json": challenge_out / "job-result.json",
            trial / "config.json": challenge_out / "config.json",
            trial / "lock.json": challenge_out / "lock.json",
            trial / "agent" / "codex.txt": challenge_out / "agent-codex.log",
            trial / "agent" / "trajectory.json": challenge_out / "agent-trajectory.json",
            trial / "trial.log": challenge_out / "trial.log",
            trial / "exception.txt": challenge_out / "exception.txt",
            trial / "artifacts" / "manifest.json": challenge_out / "artifact-manifest.json",
            CAMPAIGN / f"host-challenge-{task:02d}-{ATTEMPTS[task]}.log": challenge_out / "host-run.log",
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
            "docker_image_id": IMAGE_ID,
            "tensorcircuit_version": TC_VERSION,
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
    input_tokens = sum(row["n_input_tokens"] or 0 for row in task_rows)
    cache_tokens = sum(row["n_cache_tokens"] or 0 for row in task_rows)
    output_tokens = sum(row["n_output_tokens"] or 0 for row in task_rows)
    noncache_tokens = max(input_tokens - cache_tokens, 0)
    # Archived xAI prices for Grok 4.6: $2/M uncached input,
    # $0.50/M cached input, and $6/M output.
    list_price_cost = (
        2.0 * noncache_tokens + 0.5 * cache_tokens + 6.0 * output_tokens
    ) / 1_000_000
    summary = {
        "schema_version": 1,
        "protocol": {
            "base_commit": manifest["base_commit"],
            "solver_model": "grok-4.6",
            "solver_reasoning_effort": "high",
            "audit_model": "gpt-5.6-sol",
            "audit_reasoning_effort": "high",
            "framework": "tensorcircuit",
            "docker_image": "challenge-benchmark-quantum-tensorcircuit:py311",
            "docker_image_id": IMAGE_ID,
            "tensorcircuit_version": TC_VERSION,
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
            "input_tokens": input_tokens,
            "cache_tokens": cache_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(sum(row["cost_usd"] or 0 for row in task_rows), 6),
            "list_price_cost_usd": round(list_price_cost, 6),
            "list_price_basis_usd_per_million": {
                "uncached_input": 2.0,
                "cached_input": 0.5,
                "output": 6.0,
            },
        },
        "tasks": task_rows,
        "excluded_attempts": {},
        "notes": {
            "runtime": "runtime_score is retained for reporting and does not multiply the pass reward.",
            "selection": "Each task contributes one valid non-infrastructure outcome; infrastructure-only attempts are excluded with provenance.",
        },
    }
    dump(OUT / "summary.json", summary)
    comparison = load(OUT / "model-comparison.json")
    grok_run = comparison["runs"][-1]
    if grok_run["solver_model"] != "grok-4.6":
        raise RuntimeError("Grok 4.6 comparison row is missing")
    grok_run["passes"] = [row["challenge"] for row in task_rows if row["reward"] == 1]
    dump(OUT / "model-comparison.json", comparison)

    resources = load(OUT / "resource-comparison.json")
    grok_resources = next(
        run for run in resources["runs"] if run["label"] == "Grok 4.6 high"
    )
    grok_resources["passes"] = summary["counts"]["passes"]
    grok_resources["agent_wall_sec"] = summary["agent_usage"]["wall_time_sec"]
    grok_resources["total_cost_usd"] = summary["agent_usage"]["list_price_cost_usd"]
    grok_resources["source"] = "results/grok-4.6-high/summary.json; xAI list-price reconstruction"
    dump(OUT / "resource-comparison.json", resources)


if __name__ == "__main__":
    main()
