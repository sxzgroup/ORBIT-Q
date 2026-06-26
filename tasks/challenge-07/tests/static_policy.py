from __future__ import annotations

import ast
import json
import re
from pathlib import Path

FRAMEWORK_IMPORTS = {
    "pennylane": {"pennylane"},
    "pennylane-lane": {"pennylane"},
    "tensorcircuit": {"tensorcircuit"},
    "tensorcircuit-ng": {"tensorcircuit"},
    "tc": {"tensorcircuit"},
}

CHEATING_PATTERNS = [
    r"open\(['\"]/(solution|tests|logs)\b",
    r"Path\(['\"]/(solution|tests|logs)\b",
    r"reward\.(txt|json).*write",
    r"write_text\(.*reward\.(txt|json)",
    r"solution_\d+\.py.*read_text",
]

RAW_SIMULATOR_HINTS = [
    r"np\.kron",
    r"numpy\.kron",
    r"eigvalsh",
    r"eigh",
    r"statevector",
    r"2\s*\*\*\s*n_qubits",
    r"1\s*<<\s*n_qubits",
]


def _import_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _effective_code_lines(source: str) -> int:
    count = 0
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        count += 1
    return count


def check_source(path: Path, framework: str, max_lines: int = 200) -> dict:
    source = path.read_text()
    result = {
        "path": str(path),
        "framework": framework,
        "exists": path.exists(),
        "parse_ok": 0.0,
        "line_count": _effective_code_lines(source),
        "line_count_score": 0.0,
        "framework_score": 0.0,
        "no_forbidden_framework_score": 1.0,
        "no_static_cheating_score": 0.0,
        "no_raw_simulator_bypass_score": 1.0,
        "imports": [],
        "forbidden_imports": [],
        "cheating_hits": [],
        "raw_simulator_hits": [],
    }

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        result["syntax_error"] = str(exc)
        return result

    result["parse_ok"] = 1.0
    imports = _import_roots(tree)
    result["imports"] = sorted(imports)
    required = FRAMEWORK_IMPORTS.get(framework.lower(), {framework.lower()})

    result["line_count_score"] = 1.0 if result["line_count"] <= max_lines else 0.0
    result["framework_score"] = 1.0 if imports & required else 0.0

    cheating_hits = [
        pattern for pattern in CHEATING_PATTERNS if re.search(pattern, source, re.I)
    ]
    result["cheating_hits"] = cheating_hits
    result["no_static_cheating_score"] = 0.0 if cheating_hits else 1.0

    raw_hits = [
        pattern for pattern in RAW_SIMULATOR_HINTS if re.search(pattern, source)
    ]
    result["raw_simulator_hits"] = raw_hits
    if raw_hits and not imports & required:
        result["no_raw_simulator_bypass_score"] = 0.0
    elif len(raw_hits) >= 4:
        result["no_raw_simulator_bypass_score"] = 0.7

    components = [
        result["parse_ok"],
        result["line_count_score"],
        result["framework_score"],
        result["no_forbidden_framework_score"],
        result["no_static_cheating_score"],
        result["no_raw_simulator_bypass_score"],
    ]
    result["static_policy_score"] = float(min(components))
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--framework", default="tensorcircuit")
    parser.add_argument("--max-lines", type=int, default=200)
    args = parser.parse_args()

    print(
        json.dumps(check_source(args.source, args.framework, args.max_lines), indent=2)
    )


if __name__ == "__main__":
    main()
