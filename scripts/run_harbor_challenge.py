#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HARBOR = ROOT / ".conda" / "harbor-py312" / "bin" / "harbor"
SOLVER_AGENT_IMPORTS = {
    "codex-para": "adapters.codex_para:CodexPara",
    "claude-code": "adapters.claude_para:ClaudePara",
}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9_.-]+", "-", value.lower()).strip("-")
    if not slug:
        raise ValueError("framework slug cannot be empty")
    return slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one Harbor challenge with a command-line-selected quantum "
            "framework without modifying canonical tasks/challenge-* files."
        )
    )
    parser.add_argument(
        "--challenge",
        default=os.environ.get("CHALLENGE_ID", "01"),
        help="Challenge id such as 01, 1, or challenge-01.",
    )
    parser.add_argument(
        "--framework",
        default=os.environ.get("FRAMEWORK", "tensorcircuit"),
        help="Required quantum framework/prompt slug.",
    )
    parser.add_argument(
        "--docker-image",
        default=os.environ.get("DOCKER_IMAGE"),
        help="Framework image tag. Defaults to challenge-benchmark-quantum-<framework>:py311.",
    )
    parser.add_argument(
        "--extra-instruction-path",
        type=Path,
        default=(
            Path(os.environ["EXTRA_INSTRUCTION_PATH"])
            if os.environ.get("EXTRA_INSTRUCTION_PATH")
            else None
        ),
        help="Extra framework prompt. Defaults to prompts/frameworks/<framework>.md.",
    )
    parser.add_argument(
        "--solver-agent",
        choices=sorted(SOLVER_AGENT_IMPORTS),
        default=os.environ.get("SOLVER_AGENT", "codex-para"),
        help="Agent CLI used to solve the task. The verifier still uses CodexPara.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_NAME") or os.environ.get("MODEL"),
        help="Agent model passed to Harbor -m.",
    )
    parser.add_argument(
        "--audit-model",
        default=os.environ.get("AUDIT_MODEL_NAME") or os.environ.get("AUDIT_MODEL"),
        help="Verifier Codex audit model. Defaults to AWS-GPT-5.5.",
    )
    parser.add_argument(
        "--solver-reasoning-effort",
        default=os.environ.get("SOLVER_REASONING_EFFORT")
        or os.environ.get("CLAUDE_CODE_EFFORT_LEVEL"),
        help=(
            "Agent reasoning effort kwarg. For claude-code, defaults to max; "
            "Harbor's current ClaudeCode adapter accepts low/medium/high/xhigh/max."
        ),
    )
    parser.add_argument(
        "--job-name",
        default=os.environ.get("JOB_NAME"),
        help="Harbor job name. Defaults to challenge-<id>-<framework>-<solver-agent>.",
    )
    parser.add_argument(
        "-n",
        "--n-concurrent",
        type=int,
        default=int(os.environ.get("N_CONCURRENT", "1")),
        help="Harbor concurrency.",
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=Path(os.environ.get("JOBS_DIR", ROOT / "jobs")),
        help="Harbor jobs output directory.",
    )
    parser.add_argument(
        "--harbor-bin",
        type=Path,
        default=Path(os.environ.get("HARBOR_BIN", DEFAULT_HARBOR)),
        help="Harbor executable.",
    )
    parser.add_argument(
        "--yes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --yes to Harbor.",
    )
    return parser.parse_args()


def challenge_name(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("challenge-"):
        return raw
    return f"challenge-{int(raw):02d}"


def resolve_solver_model(solver_agent: str, explicit_model: str | None) -> str:
    if explicit_model:
        return explicit_model
    if solver_agent == "claude-code":
        model = os.environ.get("ANTHROPIC_MODEL") or os.environ.get(
            "ANTHROPIC_DEFAULT_OPUS_MODEL"
        )
        if not model:
            raise ValueError(
                "Claude solver model is required. Pass --model or export "
                "MODEL_NAME, MODEL, ANTHROPIC_MODEL, or ANTHROPIC_DEFAULT_OPUS_MODEL."
            )
        return model
    return "AWS-GPT-5.5"


def main() -> int:
    args = parse_args()
    framework = args.framework.strip()
    framework_slug = slugify(framework)
    docker_image = args.docker_image or (
        f"challenge-benchmark-quantum-{framework_slug}:py311"
    )
    challenge = challenge_name(args.challenge)
    task_dir = ROOT / "tasks" / challenge
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Canonical task not found: {task_dir}")

    extra_instruction_path = args.extra_instruction_path or (
        ROOT / "prompts" / "frameworks" / f"{framework_slug}.md"
    )
    if not extra_instruction_path.is_file():
        raise FileNotFoundError(f"Framework prompt not found: {extra_instruction_path}")

    solver_agent = args.solver_agent
    solver_model = resolve_solver_model(solver_agent, args.model)
    job_name = args.job_name or f"{challenge}-{framework_slug}-{solver_agent}"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    audit_model = args.audit_model or "AWS-GPT-5.5"
    solver_reasoning_effort = args.solver_reasoning_effort
    if solver_agent == "claude-code" and not solver_reasoning_effort:
        solver_reasoning_effort = "max"

    cmd = [
        str(args.harbor_bin),
        "run",
        "-p",
        str(task_dir),
        "--extra-instruction-path",
        str(extra_instruction_path),
        "--environment-import-path",
        "adapters.framework_docker:FrameworkDockerEnvironment",
        "--environment-kwarg",
        f"framework={framework_slug}",
        "--environment-kwarg",
        f"docker_image={docker_image}",
        "--agent-import-path",
        SOLVER_AGENT_IMPORTS[solver_agent],
        "--verifier-import-path",
        "adapters.codex_para_verifier:CodexParaVerifier",
        "--verifier-kwarg",
        f"audit_model={audit_model}",
        "--verifier-env",
        f"REQUIRED_QUANTUM_FRAMEWORK={framework_slug}",
        "-m",
        solver_model,
        "-n",
        str(args.n_concurrent),
        "-o",
        str(args.jobs_dir),
        "--job-name",
        job_name,
    ]
    if solver_reasoning_effort:
        cmd.extend(["--agent-kwarg", f"reasoning_effort={solver_reasoning_effort}"])
    if args.yes:
        cmd.append("--yes")

    print(f"Canonical task: {task_dir.relative_to(ROOT)}")
    print(f"Framework: {framework_slug}")
    print(f"Docker image: {docker_image}")
    print(f"Extra prompt: {extra_instruction_path.relative_to(ROOT)}")
    print(f"Solver agent: {solver_agent}")
    print(f"Solver model: {solver_model}")
    print(f"Solver reasoning effort: {solver_reasoning_effort or '(agent default)'}")
    print(f"Audit model: {audit_model}")
    sys.stdout.flush()
    return subprocess.run(cmd, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
