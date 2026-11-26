"""Replay one published pair from the pinned base in fresh isolated containers.

Example: python replay_task.py 04 --repo /path/to/ORBIT-Q --context colima
The exact framework image in execution-protocol.json must already be available.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess as sp
from tempfile import TemporaryDirectory


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", type=int, choices=range(1, 13))
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--context", default="colima")
    args = ap.parse_args()
    n = args.task
    root = Path(__file__).resolve().parent
    proto = json.loads((root / "execution-protocol.json").read_text())
    ev = json.loads((root / "timing-evidence.json").read_text())[str(n)]
    if "candidate" not in ev:
        ap.error("No valid candidate for this task")
    row = ev["candidate"]["stamp"]
    tag = proto["image_id"]
    docker = ["docker", "--context", args.context]
    if sp.check_output([*docker, "ps", "-q"]).strip():
        ap.error("Other containers are active")
    if sp.check_output([*docker, "image", "inspect", tag, "--format", "{{.Id}}"], text=True).strip() != tag:
        ap.error("Wrong image")
    digest = lambda b: hashlib.sha256(b).hexdigest()
    harness = root / "timing_entry.py"
    assert digest(harness.read_bytes()) == row["harness"]
    with TemporaryDirectory(prefix="orbitq-replay-") as scratch:
        scratch = Path(scratch)
        for k in ("evaluator", "candidate", "expert", "fixture", "out-expert", "out-candidate"):
            (scratch / k).mkdir()
        for sub, path, expected in [
            ("evaluator", f"tests/evaluate_{n}.py", row["evaluator"]),
            ("expert", f"solution/solution_{n}.py", ev["expert"]["stamp"]["candidate"]),
        ]:
            b = sp.check_output(["git", "-C", str(args.repo), "show", f"{proto['base']}:tasks/challenge-{n:02d}/{path}"])
            assert digest(b) == expected
            (scratch / sub / Path(path).name).write_bytes(b)
        candidate = (root / f"challenge-{n:02d}" / f"solution_{n}.py").read_bytes()
        assert digest(candidate) == row["candidate"]
        (scratch / "candidate" / f"solution_{n}.py").write_bytes(candidate)

        def execute(role):
            cmd = [
                *docker,
                "run",
                "--rm",
                "--network",
                "none",
                "--cpus",
                "6",
                "--memory",
                "10g",
                "--memory-swap",
                "10g",
                "--security-opt",
                "no-new-privileges=true",
                "--workdir",
                "/root",
            ]
            for k, val in row["threads"].items():
                cmd += ["--env", f"{k}={val}"]
            mounts = [(harness, "/entry.py", True), (scratch / "evaluator", "/evaluator", True)]
            if role == "fixture":
                mounts.append((scratch / "fixture", "/output", False))
            else:
                mounts.extend([(scratch / role, "/candidate", True), (scratch / ("out-" + role), "/output", False)])
                if n in (1, 12):
                    mounts.append((scratch / "fixture", "/fixture", True))
            for src, dst, ro in mounts:
                cmd += ["--mount", f"type=bind,src={src},dst={dst}" + (",readonly" if ro else "")]
            run = sp.run(
                [*cmd, tag, "timeout", "1200", "python", "/entry.py", "fixture" if role == "fixture" else "evaluate", str(n)],
                stdout=sp.PIPE,
                stderr=sp.STDOUT,
                text=True,
                errors="replace",
            )
            print(run.stdout, flush=True)
            if run.returncode or role == "fixture":
                return run.returncode
            return 0 if re.findall(r"^Overall:\s*(PASS|FAIL)\s*$", run.stdout, re.M) == ["PASS"] else 1

        if n in (1, 12) and execute("fixture"):
            raise SystemExit("Input fixture failed")
        codes = {}
        for role in ("expert", "candidate"):
            print(f"Task {n:02d}: {role}", flush=True)
            codes[role] = execute(role)
        if any(codes.values()):
            raise SystemExit(f"Replay failed: {codes}")


if __name__ == "__main__":
    main()
