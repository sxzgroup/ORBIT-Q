#!/usr/bin/env python3
"""Disable TorchQuantum's eager IBM runtime import inside the image.

TorchQuantum 0.1.8 imports qiskit_ibm_runtime at module import time via
torchquantum.util.utils. On Linux aarch64 inside Docker Desktop, that import can
trip an Illegal Instruction through cryptography. Benchmark tasks only need the
local simulation path, so replace the eager import with a stub that fails only
if IBM runtime access is explicitly requested.
"""

from __future__ import annotations

import sys
import sysconfig
from pathlib import Path

OLD = "from qiskit_ibm_runtime import QiskitRuntimeService\n"
CURRENT_PATCH = """try:
    from qiskit_ibm_runtime import QiskitRuntimeService
except Exception as qiskit_runtime_import_error:
    class QiskitRuntimeService:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "qiskit_ibm_runtime is unavailable in this environment"
            ) from qiskit_runtime_import_error
"""
NEW = """class QiskitRuntimeService:  # type: ignore[override]
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "qiskit_ibm_runtime is disabled in this image because its Linux "
            "aarch64 dependency chain triggers Illegal Instruction under "
            "Docker Desktop. Local TorchQuantum simulation remains available."
        )
"""


def main() -> int:
    purelib = Path(sysconfig.get_paths()["purelib"])
    target = purelib / "torchquantum" / "util" / "utils.py"

    if not target.is_file():
        print(f"No torchquantum install found at: {target}")
        return 0

    text = target.read_text()
    if NEW in text:
        print(f"Already patched: {target}")
        return 0

    if CURRENT_PATCH in text:
        target.write_text(text.replace(CURRENT_PATCH, NEW, 1))
        print(f"Updated legacy patch in: {target}")
        return 0

    if OLD not in text:
        print(f"Expected import line not found in: {target}", file=sys.stderr)
        return 1

    target.write_text(text.replace(OLD, NEW, 1))
    print(f"Patched optional qiskit_ibm_runtime import in: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
