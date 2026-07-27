from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_harbor_challenge as runner


RUNNER_ENV_NAMES = (
    "AUDIT_MODEL",
    "AUDIT_MODEL_NAME",
    "CHALLENGE_ID",
    "CODEX_AUTH_JSON_PATH",
    "CODEX_FORCE_AUTH_JSON",
    "CODEX_MODEL_CATALOG_PATH",
    "CODEX_PROFILE",
    "CODEX_PROFILE_CONFIG_PATH",
    "FORGECODE_AGENT",
    "FORGECODE_CREDENTIALS_PATH",
    "FORGECODE_MODEL",
    "FORGECODE_REASONING_EFFORT",
    "FRAMEWORK",
    "HARBOR_BIN",
    "HARBOR_YES",
    "JOB_NAME",
    "MODEL",
    "MODEL_NAME",
    "SOLVER_AGENT",
    "SOLVER_REASONING_EFFORT",
)


@pytest.fixture
def clean_runner_env(monkeypatch: pytest.MonkeyPatch):
    for name in RUNNER_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def run_and_capture(
    monkeypatch: pytest.MonkeyPatch,
    *extra_args: str,
) -> tuple[list[str], dict[str, str]]:
    captured: dict[str, object] = {}

    def fake_run(cmd, *, env, check):
        captured["cmd"] = cmd
        captured["env"] = env
        captured["check"] = check
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            "run_harbor_challenge.py",
            "--challenge",
            "01",
            "--framework",
            "tensorcircuit",
            "--harbor-bin",
            "harbor",
            "--no-yes",
            *extra_args,
        ],
    )

    assert runner.main() == 0
    assert captured["check"] is False
    return captured["cmd"], captured["env"]


def test_forgecode_uses_provider_model_and_codex_login(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_runner_env,
):
    auth_json = tmp_path / "auth.json"
    auth_json.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CODEX_AUTH_JSON_PATH", str(auth_json))

    cmd, _ = run_and_capture(
        monkeypatch,
        "--solver-agent",
        "forgecode",
        "--model",
        "codex/gpt-5.6-sol",
        "--solver-reasoning-effort",
        "xhigh",
        "--codex-force-auth-json",
    )

    assert "adapters.forgecode:ForgeCode" in cmd
    assert "reasoning_effort=xhigh" in cmd
    assert f"codex_auth_json_path={auth_json}" in cmd
    assert "agent_id=forge" in cmd


def test_forgecode_accepts_its_credentials_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_runner_env,
):
    credentials = tmp_path / ".credentials.json"
    credentials.write_text("[]", encoding="utf-8")

    cmd, _ = run_and_capture(
        monkeypatch,
        "--solver-agent",
        "forgecode",
        "--model",
        "anthropic/claude-sonnet-4-6",
        "--forgecode-credentials-path",
        str(credentials),
        "--forgecode-agent",
        "forge",
    )

    assert f"credentials_path={credentials}" in cmd
    assert "agent_id=forge" in cmd


def test_forgecode_rejects_model_without_provider(
    monkeypatch: pytest.MonkeyPatch,
    clean_runner_env,
):
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            "run_harbor_challenge.py",
            "--challenge",
            "01",
            "--solver-agent",
            "forgecode",
            "--model",
            "gpt-5.6-sol",
        ],
    )

    with pytest.raises(ValueError, match="provider/model"):
        runner.main()
