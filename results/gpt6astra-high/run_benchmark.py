#!/usr/bin/env python3
"""Prepare fresh task copies and run the twelve historical Harbor trials."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/gpt6astra-high"
RUN = ROOT / ".tmp/astra-high-20260905"
HOST_REPO = Path("/Users/qqy/Desktop/2026Project/ORBIT-Q")
HARBOR = HOST_REPO / ".conda/harbor-py312/bin/harbor"
JOBS = HOST_REPO / "jobs/gpt6astra-high-clean-20260905"
BASE = "0201238ec2983907e2891f5319f5fff2d00844d5"
IMAGE = "challenge-benchmark-quantum-tensorcircuit:astra-cli01533-audit01450"
PARENT_IMAGE = "sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833"


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


def prepare():
    if RUN.exists():
        raise RuntimeError("Execution task directory already exists; refusing to replace it")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(["git", "diff", "--exit-code", BASE, "--", "tasks", "adapters", "prompts"],
                   cwd=ROOT, check=True)
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--", "tasks", "adapters", "prompts"], cwd=ROOT, text=True)
    assert not untracked.strip(), "Untracked files in benchmark inputs"
    task_hashes, comparisons = {}, []
    for n in range(1, 13):
        name = f"challenge-{n:02d}"
        task = RUN / name
        shutil.copytree(ROOT / "tasks" / name, task)
        config = task / "task.toml"
        original = config.read_text()
        assert original.count("cpus = 14") == 1
        assert original.count("memory_mb = 32768") == 1
        config.write_text(original.replace("cpus = 14", "cpus = 6")
                         .replace("memory_mb = 32768", "memory_mb = 10240"))
        assert (task / "environment").is_dir()
        task_hashes[name] = hashes(task)
        # Host-only byte comparison; previous copies are never supplied to agents.
        old = HOST_REPO / ".scratch/gpt56sol-ultra-20260728" / name
        comparisons.append({"challenge": n, "identical_to_previous_execution_task":
                            hashes(old) == task_hashes[name] if old.is_dir() else None})
    manifest = {
        "created_at": now(), "branch": "codex/gpt-6-astra-high-clean-benchmark",
        "base_commit": BASE, "execution_checkout_commit": head,
        "solver_model": "gpt-6-astra", "solver_effort": "high",
        "audit_model": "gpt-5.6-sol", "audit_effort": "high",
        "solver_cli": "0.153.3", "audit_cli": "0.145.0", "harbor_version": "0.20.0",
        "docker_image": IMAGE, "parent_image_id": PARENT_IMAGE,
        "cpus": 6, "memory_mb": 10240, "storage_mb": 16384,
        "trials_per_challenge": 1, "harbor_retries": 0, "concurrency": 1,
        "execution_order": list(range(1, 13)), "task_hashes": task_hashes,
        "previous_task_comparison": comparisons,
        "solver_inputs": ["instruction.md", "prompts/frameworks/tensorcircuit.md", "installed image APIs"],
        "resume": False, "skills": [], "mcp_servers": [], "extra_host_mounts": [],
        "cli_change_reason": "Service rejects gpt-6-astra on Codex 0.145.0; solver uses 0.153.3. Audit uses original 0.145.0.",
        "comparison_scope": "Astra high vs Sol high changes model; Sol high vs Sol ultra changes effort. No pure Astra effort effect is estimated from one effort level.",
    }
    dump(OUT / "protocol.json", manifest)
    print(json.dumps({"prepared_tasks": 12, "previous_task_comparison": comparisons}), flush=True)


def command(n):
    task = RUN / f"challenge-{n:02d}"
    job = f"challenge-{n:02d}-tensorcircuit-gpt-6-astra-high-clean-r1"
    cmd = [str(HARBOR), "run", "-p", str(task),
           "--extra-instruction-path", str(ROOT / "prompts/frameworks/tensorcircuit.md"),
           "--environment-import-path", "adapters.framework_docker:FrameworkDockerEnvironment",
           "--environment-kwarg", "framework=tensorcircuit",
           "--environment-kwarg", f"docker_image={IMAGE}",
           "--agent-import-path", "harbor.agents.installed.codex:Codex",
           "--agent-kwarg", "reasoning_effort=high",
           "--agent-env", "CODEX_FORCE_AUTH_JSON=true",
           "--verifier-import-path", "adapters.codex_para_verifier:CodexParaVerifier",
           "--verifier-kwarg", "audit_model=gpt-5.6-sol",
           "--verifier-kwarg", "force_auth_json=true",
           "--verifier-kwarg", f"profile_config_path={OUT / 'audit-high.config.toml'}",
           "--verifier-env", "REQUIRED_QUANTUM_FRAMEWORK=tensorcircuit",
           "--verifier-env", "PATH=/usr/local/bin:/usr/bin:/bin",
           "-m", "gpt-6-astra", "-n", "1", "-k", "1", "--max-retries", "0",
           "-o", str(JOBS), "--job-name", job, "--yes"]
    proxy = os.environ.get("BENCHMARK_PROXY", "http://192.168.5.2:7891")
    for flag in ["--agent-env", "--verifier-env"]:
        cmd.extend([flag, f"HTTP_PROXY={proxy}", flag, f"HTTPS_PROXY={proxy}",
                    flag, "NO_PROXY=localhost,127.0.0.1"])
    return cmd, job


def archive(n, job):
    jobdir = JOBS / job
    dest = OUT / f"challenge-{n:02d}"
    dest.mkdir(exist_ok=True)
    result = jobdir / "result.json"
    if result.exists():
        shutil.copy2(result, dest / "job-result.json")
    trials = sorted(jobdir.glob(f"challenge-{n:02d}__*"))
    if len(trials) != 1:
        return {"challenge": n, "status": "incomplete", "trial_count": len(trials)}
    trial = trials[0]
    for source, target in [
        (trial / "artifacts/root" / f"solution_{n}.py", dest / f"solution_{n}.py"),
        (trial / "verifier/reward.json", dest / "reward.json"),
        (trial / "verifier/audit-details.json", dest / "audit-details.json"),
        (trial / "verifier/functional.stdout", dest / "functional-stdout-official.txt"),
        (trial / "result.json", dest / "trial-result.json"),
    ]:
        if source.is_file():
            shutil.copy2(source, target)
    reward_path = dest / "reward.json"
    reward = json.loads(reward_path.read_text()) if reward_path.exists() else {}
    return {"challenge": n, "trial": trial.name, "status": "finished",
            **{k: reward.get(k) for k in ["reward", "functional_score", "static_policy_score", "llm_audit_score", "runtime_sec"]}}


def run(numbers):
    protocol = json.loads((OUT / "protocol.json").read_text())
    image_id = subprocess.check_output(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True).strip()
    protocol["docker_image_id"] = image_id
    dump(OUT / "protocol.json", protocol)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    # No private profile, endpoint, catalog, or earlier trajectory is inherited.
    for key in ["OPENAI_BASE_URL", "CODEX_PROFILE", "CODEX_PROFILE_CONFIG_PATH", "CODEX_MODEL_CATALOG_PATH", "CODEX_AUTH_JSON_PATH"]:
        env.pop(key, None)
    env["CODEX_FORCE_AUTH_JSON"] = "true"
    for n in numbers:
        name = f"challenge-{n:02d}"
        assert hashes(RUN / name) == protocol["task_hashes"][name]
        cmd, job = command(n)
        if (JOBS / job).exists():
            raise RuntimeError(f"Refusing to reuse existing job {job}")
        print(f"[{now()}] START {name}", flush=True)
        dump(RUN / f"{name}-command.json", cmd)
        with (RUN / f"host-{name}.log").open("w") as log:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            dump(RUN / "active.json", {"challenge": n, "pid": process.pid, "job": job, "started_at": now()})
            rc = process.wait()
        row = archive(n, job)
        row.update({"harbor_exit_code": rc, "finished_at": now()})
        dump(OUT / name / "status.json", row)
        print(json.dumps(row), flush=True)
        # Do not consume all remaining trials if the service/model is unavailable.
        if rc != 0 and row.get("functional_score") is None:
            raise RuntimeError(f"Infrastructure failure for {name}; inspect logs before proceeding")
    dump(RUN / "active.json", {"status": "completed", "finished_at": now()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run", "archive"])
    parser.add_argument("challenges", nargs="*", type=int)
    args = parser.parse_args()
    numbers = args.challenges or list(range(1, 13))
    assert all(1 <= n <= 12 for n in numbers)
    if args.mode == "prepare":
        prepare()
    elif args.mode == "run":
        run(numbers)
    else:
        for n in numbers:
            print(json.dumps(archive(n, command(n)[1])))
