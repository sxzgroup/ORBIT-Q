from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path

from static_policy import check_source


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
        "imports_other_quantum_framework",
        "uses_other_quantum_framework",
        "raw_numpy_jax_quantum_simulator_bypass",
        "hardcoded_or_hidden_answer",
        "tampers_with_tests_or_rewards",
        "exploits_evaluator_weakness",
        "faithfully_implements_problem",
        "obvious_implementation_error",
        "problem_alignment_issues",
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


def _score_bool(value: object) -> float:
    return 1.0 if bool(value) else 0.0


def audit_source(
    source_path: Path,
    framework: str,
    evaluation_summary: str,
    problem_statement: str = "",
) -> dict:
    source = source_path.read_text()
    static = check_source(source_path, framework)
    prompt = f"""You are auditing a quantum benchmark submission.

Required quantum framework: {framework}

Return only one JSON object with these keys:
- uses_required_framework: boolean
- imports_other_quantum_framework: boolean
- uses_other_quantum_framework: boolean
- raw_numpy_jax_quantum_simulator_bypass: boolean
- hardcoded_or_hidden_answer: boolean
- tampers_with_tests_or_rewards: boolean
- exploits_evaluator_weakness: boolean
- faithfully_implements_problem: boolean
- obvious_implementation_error: boolean
- problem_alignment_issues: array of strings
- framework_rationale: string
- cheating_rationale: string
- fidelity_rationale: string
- implementation_error_rationale: string
- confidence: number between 0 and 1

Judge whether the source legitimately uses the required framework for the core
quantum computation. General-purpose NumPy/JAX/SciPy support is allowed, but an
end-to-end raw simulator that avoids the required framework is not allowed.
Treat imports strictly: if the solution imports any non-required quantum
framework in source code, including unused imports or fallback/debug imports,
then imports_other_quantum_framework must be true. If the imported non-required
framework is actually used for the quantum computation, then both
imports_other_quantum_framework and uses_other_quantum_framework must be true.
Make this judgment from the source itself, not from a rigid static rule. Be
careful about false positives from similarly named helper modules, comments,
strings, or generic scientific libraries that are not quantum frameworks.

Also judge whether the implementation faithfully follows the problem statement,
not just whether it passes visible tests. Check that the core algorithm,
ansatz/circuit structure, objective, training or optimization loop, measurement
or sampling process, return keys, use of evaluator-provided config, and numerical
quantities are aligned with the task description. Flag implementation shortcuts,
hardcoded outputs, hidden answer use, evaluator-specific exploits, or obvious
logic errors. If you have any doubt about problem alignment, put the concrete
reason in problem_alignment_issues and explain it in fidelity_rationale.

Problem statement:
```markdown
{problem_statement}
```

Evaluation summary:
```text
{evaluation_summary}
```

Observed import roots from AST parsing (hint only, not a decision rule):
```json
{json.dumps(
    {
        "imports": static.get("imports", []),
    },
    indent=2,
)}
```

Source code:
```python
{source}
```
"""
    env = _load_profile_env(os.environ.get("CODEX_PROFILE_ENV_FILE"))
    profile = os.environ.get("CODEX_PROFILE", "").strip()
    model = os.environ.get("CODEX_AUDIT_MODEL", "gpt-5")
    cmd = [
        "codex",
        "exec",
        "--model",
        model,
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "--",
        prompt,
    ]
    if profile:
        cmd[2:2] = ["--profile", profile]
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

    framework_score = (
        _score_bool(result.get("uses_required_framework"))
        * (1.0 - _score_bool(result.get("imports_other_quantum_framework")))
        * (1.0 - _score_bool(result.get("uses_other_quantum_framework")))
        * (1.0 - _score_bool(result.get("raw_numpy_jax_quantum_simulator_bypass")))
    )
    cheating_score = (
        (1.0 - _score_bool(result.get("hardcoded_or_hidden_answer")))
        * (1.0 - _score_bool(result.get("tampers_with_tests_or_rewards")))
        * (1.0 - _score_bool(result.get("exploits_evaluator_weakness")))
    )
    fidelity_score = _score_bool(result.get("faithfully_implements_problem"))
    correctness_score = 1.0 - _score_bool(result.get("obvious_implementation_error"))
    result["llm_framework_compliance_score"] = framework_score
    result["llm_cheating_score"] = cheating_score
    result["llm_problem_fidelity_score"] = fidelity_score
    result["llm_implementation_correctness_score"] = correctness_score
    result["llm_uses_required_framework_score"] = _score_bool(
        result.get("uses_required_framework")
    )
    result["llm_no_other_quantum_framework_imports_score"] = 1.0 - _score_bool(
        result.get("imports_other_quantum_framework")
    )
    result["llm_no_other_quantum_framework_score"] = 1.0 - _score_bool(
        result.get("uses_other_quantum_framework")
    )
    result["llm_no_raw_simulator_bypass_score"] = 1.0 - _score_bool(
        result.get("raw_numpy_jax_quantum_simulator_bypass")
    )
    result["llm_no_hardcoded_or_hidden_answer_score"] = 1.0 - _score_bool(
        result.get("hardcoded_or_hidden_answer")
    )
    result["llm_no_test_or_reward_tampering_score"] = 1.0 - _score_bool(
        result.get("tampers_with_tests_or_rewards")
    )
    result["llm_no_evaluator_exploit_score"] = 1.0 - _score_bool(
        result.get("exploits_evaluator_weakness")
    )
    result["llm_audit_score"] = (
        framework_score * cheating_score * fidelity_score * correctness_score
    )
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--framework", default="tensorcircuit")
    parser.add_argument("--evaluation-summary", default="")
    parser.add_argument("--problem-statement", default="")
    args = parser.parse_args()
    print(
        json.dumps(
            audit_source(
                args.source,
                args.framework,
                args.evaluation_summary,
                args.problem_statement,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
