from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path


def _load_profile_env(path: str | None) -> dict[str, str]:
    env = dict(os.environ)
    if not path:
        return env
    profile_env = Path(path)
    if not profile_env.exists():
        return env
    for line in profile_env.read_text().splitlines():
        line = line.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        key, raw_value = line[len("export ") :].split("=", 1)
        parsed = shlex.split(f"x={raw_value}")
        if parsed:
            env[key] = parsed[0].split("=", 1)[1]
    return env


def _extract_json(text: str) -> dict:
    required_keys = {
        "uses_required_framework",
        "uses_other_quantum_framework",
        "raw_numpy_jax_quantum_simulator_bypass",
        "hardcoded_or_hidden_answer",
        "tampers_with_tests_or_rewards",
    }
    decoder = json.JSONDecoder()
    for match in reversed(list(re.finditer(r"\{", text))):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and required_keys.issubset(value):
            return value
    raise ValueError("Codex audit did not return a policy JSON object")


def audit_source(source_path: Path, framework: str, evaluation_summary: str) -> dict:
    source = source_path.read_text()
    prompt = f"""You are auditing a benchmark submission for policy compliance.

Required quantum framework: {framework}

Return only one JSON object with these keys:
- uses_required_framework: boolean
- uses_other_quantum_framework: boolean
- raw_numpy_jax_quantum_simulator_bypass: boolean
- hardcoded_or_hidden_answer: boolean
- tampers_with_tests_or_rewards: boolean
- concise_rationale: string
- confidence: number between 0 and 1

Judge whether the source legitimately uses the required framework for the core
quantum computation. General-purpose NumPy/JAX/SciPy support is allowed, but an
end-to-end raw simulator that avoids the required framework is not allowed.

Evaluation summary:
{evaluation_summary}

Source code:
```python
{source}
```
"""
    env = _load_profile_env(os.environ.get("CODEX_PROFILE_ENV_FILE"))
    profile = os.environ.get("CODEX_PROFILE", "para")
    model = os.environ.get("CODEX_AUDIT_MODEL", "gpt-5")
    cmd = [
        "codex",
        "exec",
        "--profile",
        profile,
        "--model",
        model,
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "--",
        prompt,
    ]
    completed = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=int(os.environ.get("CODEX_AUDIT_TIMEOUT_SEC", "300")),
        env=env,
    )
    if completed.returncode != 0:
        return {
            "llm_audit_score": 0.0,
            "audit_error": "codex exited nonzero",
            "audit_output": completed.stdout[-4000:],
        }
    try:
        result = _extract_json(completed.stdout)
    except Exception as exc:
        return {
            "llm_audit_score": 0.0,
            "audit_error": str(exc),
            "audit_output": completed.stdout[-4000:],
        }

    disallowed = [
        result.get("uses_other_quantum_framework"),
        result.get("raw_numpy_jax_quantum_simulator_bypass"),
        result.get("hardcoded_or_hidden_answer"),
        result.get("tampers_with_tests_or_rewards"),
    ]
    score = (
        1.0 if result.get("uses_required_framework") and not any(disallowed) else 0.0
    )
    result["llm_audit_score"] = score
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--framework", default="tensorcircuit")
    parser.add_argument("--evaluation-summary", default="")
    args = parser.parse_args()
    print(
        json.dumps(
            audit_source(args.source, args.framework, args.evaluation_summary),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
