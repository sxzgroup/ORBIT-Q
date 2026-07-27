import base64
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from adapters.forgecode import ForgeCode


def jwt_with_claims(claims: dict) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode()
    return f"header.{payload.rstrip('=')}.signature"


def test_codex_auth_is_converted_to_forgecode_credentials(tmp_path: Path):
    expires_at = 1_900_000_000
    access_token = jwt_with_claims(
        {
            "exp": expires_at,
            "https://api.openai.com/auth": {"chatgpt_account_id": "account-from-claim"},
        }
    )
    auth_json = tmp_path / "auth.json"
    auth_json.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": "refresh-token",
                },
            }
        ),
        encoding="utf-8",
    )

    credentials = ForgeCode._forge_credentials_from_codex_auth(auth_json)

    assert len(credentials) == 1
    credential = credentials[0]
    assert credential["id"] == "codex"
    assert credential["url_params"]["chatgpt_account_id"] == "account-from-claim"
    oauth = credential["auth_details"]["o_auth"]
    assert oauth["tokens"]["access_token"] == access_token
    assert oauth["tokens"]["refresh_token"] == "refresh-token"
    assert oauth["tokens"]["expires_at"] == datetime.fromtimestamp(
        expires_at, timezone.utc
    ).isoformat().replace("+00:00", "Z")
    assert oauth["config"]["client_id"].startswith("app_")


def test_forgecode_requires_provider_model():
    adapter = ForgeCode(logs_dir=Path("/tmp/logs"), model_name="gpt-5.6-sol")

    with pytest.raises(ValueError, match="provider/model"):
        adapter._split_model()


def test_forgecode_normalizes_provider_slug():
    adapter = ForgeCode(
        logs_dir=Path("/tmp/logs"),
        model_name="open-router/model-name",
    )

    assert adapter._split_model() == ("open_router", "model-name")


def test_forgecode_rejects_unsafe_agent_id():
    with pytest.raises(ValueError, match="agent_id"):
        ForgeCode(
            logs_dir=Path("/tmp/logs"),
            model_name="codex/gpt-5.6-sol",
            agent_id="forge; touch /tmp/unsafe",
        )
