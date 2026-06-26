from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from static_policy import check_source


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _runtime_score(runtime_sec: float | None) -> float:
    if runtime_sec is None:
        return 0.0
    full_score_sec = float(os.environ.get("RUNTIME_FULL_SCORE_SEC", "180"))
    zero_score_sec = float(os.environ.get("RUNTIME_ZERO_SCORE_SEC", "300"))
    if runtime_sec <= full_score_sec:
        return 1.0
    if runtime_sec >= zero_score_sec:
        return 0.0
    return float((zero_score_sec - runtime_sec) / (zero_score_sec - full_score_sec))


def _parse_runtime_sec(output: str) -> float | None:
    match = re.search(r"End-to-end solution time:\s*([0-9]+(?:\.[0-9]+)?)s", output)
    if not match:
        return None
    return float(match.group(1))


def _functional_score(problem_id: int, solution_module: str) -> dict:
    evaluate_path = Path(f"/tests/evaluate_{problem_id}.py")
    cmd = [sys.executable, str(evaluate_path), "--solution", solution_module]
    extra = os.environ.get("EVALUATE_EXTRA_ARGS", "").strip()
    if extra:
        cmd.extend(extra.split())
    env = dict(os.environ)
    env["PYTHONPATH"] = "/root:/tests" + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    try:
        completed = subprocess.run(
            cmd,
            cwd="/root",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(os.environ.get("FUNCTIONAL_TIMEOUT_SEC", "900")),
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        output += "\nFUNCTIONAL_TIMEOUT"
        Path("/logs/verifier/functional-stdout.txt").write_text(output)
        return {
            "functional_score": 0.0,
            "runtime_score": 0.0,
            "runtime_sec": -1.0,
            "functional_return_code": -1,
            "functional_passed": False,
            "functional_output_tail": output[-4000:],
        }
    output = completed.stdout
    runtime_sec = _parse_runtime_sec(output)
    passed = completed.returncode == 0 and "Overall: PASS" in output
    Path("/logs/verifier/functional-stdout.txt").write_text(output)
    return {
        "functional_score": 1.0 if passed else 0.0,
        "runtime_score": _runtime_score(runtime_sec),
        "runtime_sec": -1.0 if runtime_sec is None else runtime_sec,
        "functional_return_code": completed.returncode,
        "functional_passed": passed,
        "functional_output_tail": output[-4000:],
    }


def _llm_audit_score(
    source: Path,
    framework: str,
    evaluation_summary: str,
    problem_statement: str,
) -> dict:
    if os.environ.get("CODEX_AUDIT_ENABLED", "1") == "0":
        return {"llm_audit_score": 1.0, "llm_audit_skipped": True}
    audit_module = _load_module(Path("/tests/audit_codex.py"), "audit_codex_runtime")
    try:
        return audit_module.audit_source(
            source,
            framework,
            evaluation_summary,
            problem_statement,
        )
    except Exception as exc:
        return {"llm_audit_score": 0.0, "audit_error": str(exc)}


def main() -> None:
    problem_id = int(Path("/tests/problem_id.txt").read_text().strip())
    framework = os.environ.get("REQUIRED_QUANTUM_FRAMEWORK", "tensorcircuit")
    solution_module = f"solution_{problem_id}"
    solution_path = Path(f"/root/{solution_module}.py")

    rewards = {
        "problem_id": problem_id,
        "reward": 0.0,
        "functional_score": 0.0,
        "runtime_score": 0.0,
        "runtime_sec": -1.0,
        "static_policy_score": 0.0,
        "llm_audit_score": 0.0,
        "line_count_score": 0.0,
        "framework_score": 0.0,
        "no_forbidden_framework_score": 0.0,
        "no_raw_simulator_bypass_score": 0.0,
        "no_static_cheating_score": 0.0,
        "llm_framework_compliance_score": 0.0,
        "llm_cheating_score": 0.0,
        "llm_problem_fidelity_score": 0.0,
        "llm_implementation_correctness_score": 0.0,
        "llm_uses_required_framework_score": 0.0,
        "llm_no_other_quantum_framework_imports_score": 0.0,
        "llm_no_other_quantum_framework_score": 0.0,
        "llm_no_raw_simulator_bypass_score": 0.0,
        "llm_no_hardcoded_or_hidden_answer_score": 0.0,
        "llm_no_test_or_reward_tampering_score": 0.0,
        "llm_no_evaluator_exploit_score": 0.0,
    }
    details = {"problem_id": problem_id, "framework": framework}

    if not solution_path.exists():
        details["missing_solution"] = str(solution_path)
        Path("/logs/verifier/reward.json").write_text(json.dumps(rewards, indent=2))
        Path("/logs/verifier/audit-details.json").write_text(
            json.dumps(details, indent=2)
        )
        return

    static = check_source(solution_path, framework)
    functional = _functional_score(problem_id, solution_module)
    problem_statement_path = Path("/tests/problem_statement.md")
    problem_statement = (
        problem_statement_path.read_text(errors="replace")
        if problem_statement_path.exists()
        else ""
    )
    audit = _llm_audit_score(
        solution_path,
        framework,
        functional.get("functional_output_tail", ""),
        problem_statement,
    )

    details.update({"static": static, "functional": functional, "audit": audit})
    rewards.update(
        {
            "functional_score": float(functional["functional_score"]),
            "runtime_score": float(functional["runtime_score"]),
            "runtime_sec": float(functional["runtime_sec"]),
            "static_policy_score": float(static["static_policy_score"]),
            "llm_audit_score": float(audit.get("llm_audit_score", 0.0)),
            "line_count_score": float(static["line_count_score"]),
            "framework_score": float(static["framework_score"]),
            "no_forbidden_framework_score": float(
                static["no_forbidden_framework_score"]
            ),
            "no_raw_simulator_bypass_score": float(
                static["no_raw_simulator_bypass_score"]
            ),
            "no_static_cheating_score": float(static["no_static_cheating_score"]),
            "llm_framework_compliance_score": float(
                audit.get("llm_framework_compliance_score", 0.0)
            ),
            "llm_cheating_score": float(audit.get("llm_cheating_score", 0.0)),
            "llm_problem_fidelity_score": float(
                audit.get("llm_problem_fidelity_score", 0.0)
            ),
            "llm_implementation_correctness_score": float(
                audit.get("llm_implementation_correctness_score", 0.0)
            ),
            "llm_uses_required_framework_score": float(
                audit.get("llm_uses_required_framework_score", 0.0)
            ),
            "llm_no_other_quantum_framework_imports_score": float(
                audit.get("llm_no_other_quantum_framework_imports_score", 0.0)
            ),
            "llm_no_other_quantum_framework_score": float(
                audit.get("llm_no_other_quantum_framework_score", 0.0)
            ),
            "llm_no_raw_simulator_bypass_score": float(
                audit.get("llm_no_raw_simulator_bypass_score", 0.0)
            ),
            "llm_no_hardcoded_or_hidden_answer_score": float(
                audit.get("llm_no_hardcoded_or_hidden_answer_score", 0.0)
            ),
            "llm_no_test_or_reward_tampering_score": float(
                audit.get("llm_no_test_or_reward_tampering_score", 0.0)
            ),
            "llm_no_evaluator_exploit_score": float(
                audit.get("llm_no_evaluator_exploit_score", 0.0)
            ),
        }
    )
    rewards["reward"] = (
        rewards["functional_score"]
        * rewards["runtime_score"]
        * rewards["static_policy_score"]
        * rewards["llm_audit_score"]
    )

    Path("/root/evaluation_summary.json").write_text(json.dumps(details, indent=2))
    Path("/logs/verifier/audit-details.json").write_text(json.dumps(details, indent=2))
    Path("/logs/verifier/reward.json").write_text(json.dumps(rewards, indent=2))


if __name__ == "__main__":
    main()
