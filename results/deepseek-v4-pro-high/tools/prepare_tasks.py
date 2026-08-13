#!/usr/bin/env python3
"""Copy the frozen task set used by the Terra and Luna campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    args = parser.parse_args()

    if args.run_root.exists() and any(args.run_root.iterdir()):
        raise FileExistsError(f"run root must be empty: {args.run_root}")
    args.run_root.mkdir(parents=True, exist_ok=True)

    source_manifest = json.loads(
        (args.source_run_root / "task-copy-manifest.json").read_text()
    )
    if source_manifest["base_commit"] != args.base_commit:
        raise ValueError("source task-copy manifest uses a different base commit")

    records = []
    for source_record in source_manifest["tasks"]:
        challenge = source_record["challenge"]
        source = args.source_run_root / challenge
        destination = args.run_root / challenge
        source_hash = tree_sha256(source)
        if source_hash != source_record["execution_copy_tree_sha256"]:
            raise ValueError(f"source task copy hash mismatch: {challenge}")
        shutil.copytree(source, destination)
        if tree_sha256(destination) != source_hash:
            raise ValueError(f"copied task hash mismatch: {challenge}")
        records.append(dict(source_record))

    manifest = dict(source_manifest)
    manifest["task_copy_root"] = str(args.run_root)
    manifest["source_task_copy_root"] = str(args.source_run_root)
    manifest["tasks"] = records
    (args.run_root / "task-copy-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
