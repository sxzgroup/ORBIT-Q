"""Verify Grok 4.6/high protocol stamps, scores, hashes, and figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(text.replace("[REDACTED]", "null"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    summary = load(ROOT / "summary.json")
    rows = summary["tasks"]
    assert len(rows) == 12
    counts = summary["counts"]
    assert counts["valid_outcomes"] == 12
    assert counts["passes"] == sum(row["reward"] == 1 for row in rows)
    assert counts["functional_passes"] == sum(
        row["functional_score"] == 1 for row in rows
    )
    assert counts["static_policy_passes"] == sum(
        row["static_policy_score"] == 1 for row in rows
    )
    assert counts["audit_passes"] == sum(row["llm_audit_score"] == 1 for row in rows)

    for row in rows:
        task = row["challenge"]
        folder = ROOT / f"challenge-{task:02d}"
        reward = load(folder / "reward.json")
        stamp = load(folder / "stamp-info.json")
        config = load(folder / "config.json")
        assert reward["reward"] == row["reward"] == stamp["reward"]
        assert config["agent"]["model_name"] == "grok-4.6"
        assert config["agent"]["kwargs"]["reasoning_effort"] == "high"
        assert config["verifier"]["kwargs"]["audit_model"] == "gpt-5.6-sol"
        assert config["verifier"]["kwargs"]["profile_config_path"].endswith(
            "audit-high.config.toml"
        )
        assert row["tensorcircuit_version"] == "1.8.0.dev20260726"
        assert row["docker_image_id"] == (
            "sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833"
        )
        solution = folder / f"solution_{task}.py"
        if row["solution_sha256"] is None:
            assert not solution.exists()
        else:
            assert solution.exists()
            assert sha256(solution) == row["solution_sha256"]

    assert (ROOT / "audit-high.config.toml").read_text().strip() == (
        'model_reasoning_effort = "high"'
    )
    manifest = load(ROOT / "task-copy-manifest.json")
    assert manifest["base_commit"] == "0201238ec2983907e2891f5319f5fff2d00844d5"
    assert manifest["aggregate_task_copy_sha256"] == (
        "19fe27b83eaf668b3df32d1a68902b08cbe28585f189290769018eb16d927895"
    )

    comparison = load(ROOT / "model-comparison.json")
    pass_sets = {run["label"]: run["passes"] for run in comparison["runs"]}
    assert pass_sets["DeepSeek V4 Flash high"] == [3, 4, 5, 10, 12]
    assert pass_sets["Grok 4.5 high"] == [2, 3, 5, 6, 7, 9, 10, 12]
    assert pass_sets["DeepSeek V4 Pro high"] == [2, 4, 6, 7, 10]
    assert pass_sets["Grok 4.6 high"] == [2, 3, 4, 6, 7, 9, 10, 11, 12]

    resources = load(ROOT / "resource-comparison.json")
    resource_runs = {run["label"]: run for run in resources["runs"]}
    assert resource_runs["DeepSeek V4 Flash high"]["total_cost_usd"] == 0.572672
    assert resource_runs["Grok 4.5 high"]["total_cost_usd"] == 5.09014
    assert resource_runs["DeepSeek V4 Pro high"]["total_cost_usd"] == 1.101116
    assert resource_runs["Grok 4.6 high"]["total_cost_usd"] > 0

    for stem in ("grok-4.6-high-outcomes", "grok-4.6-high-agent-resource-use"):
        png = ROOT / "figs" / f"{stem}.png"
        assert png.stat().st_size > 50_000
        with Image.open(png) as image:
            assert image.width >= 2500 and image.height >= 800

    print(
        "Archive verification passed: 12 outcomes, Grok 4.6/high solver, "
        "Sol/high audit, fixed task hashes, and figures valid."
    )


if __name__ == "__main__":
    main()
