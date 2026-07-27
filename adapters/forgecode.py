from __future__ import annotations

import base64
import json
import re
import shlex
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, override

from harbor.agents.installed.base import (
    AgentAuthenticationError,
    BaseInstalledAgent,
    EnvVar,
    ErrorPattern,
    ModelNotFoundError,
    with_prompt_template,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

_FORGECODE_AGENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SOLUTION_RE = re.compile(r"\bsolution_(\d+)\.py\b")
_VERSION_RE = re.compile(r"(\d+\.\d+\.\d+)")

_PROVIDER_ENV_VARS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_URL"),
    "cerebras": ("CEREBRAS_API_KEY",),
    "codex": ("CODEX_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "forge": ("FORGE_API_KEY",),
    "github_copilot": ("GITHUB_COPILOT_API_KEY",),
    "google_ai_studio": ("GOOGLE_AI_STUDIO_API_KEY", "GOOGLE_API_KEY"),
    "groq": ("GROQ_API_KEY",),
    "kimi_coding": ("KIMI_API_KEY",),
    "minimax": ("MINIMAX_API_KEY",),
    "moonshot": ("MOONSHOT_API_KEY",),
    "open_router": ("OPENROUTER_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "openai_compatible": ("OPENAI_API_KEY", "OPENAI_URL"),
    "openai_responses_compatible": ("OPENAI_API_KEY", "OPENAI_URL"),
    "requesty": ("REQUESTY_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "zai": ("ZAI_API_KEY",),
    "zai_coding": ("ZAI_CODING_API_KEY",),
}

_CODEX_OAUTH_CONFIG = {
    "auth_url": "https://auth.openai.com/oauth/authorize",
    "token_url": "https://auth.openai.com/oauth/token",
    "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
    "scopes": ["openid", "profile", "email", "offline_access"],
    "redirect_uri": "http://localhost:1455/auth/callback",
    "use_pkce": True,
    "custom_headers": {"originator": "forge"},
    "extra_auth_params": {
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
    },
}


class ForgeCode(BaseInstalledAgent):
    """Harbor adapter for ForgeCode's non-interactive one-shot CLI."""

    _OUTPUT_FILENAME = "forgecode.txt"
    _REMOTE_CONFIG_DIR = Path("/tmp/harbor-forgecode")

    ENV_VARS = [
        EnvVar(
            "reasoning_effort",
            env="FORGE_REASONING__EFFORT",
            type="enum",
            choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
            env_fallback="FORGECODE_REASONING_EFFORT",
        ),
    ]
    ERROR_PATTERNS = [
        *BaseInstalledAgent.ERROR_PATTERNS,
        ErrorPattern(
            r"provider .+ (is not configured|requires credentials)|"
            r"provider .+ is not available|no configured provider|"
            r"authentication required",
            AgentAuthenticationError,
        ),
        ErrorPattern(
            r"model .+ (not found|is not available)|undefined model",
            ModelNotFoundError,
        ),
    ]

    def __init__(
        self,
        *args: Any,
        credentials_path: str | None = None,
        codex_auth_json_path: str | None = None,
        agent_id: str = "forge",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        if not agent_id or not _FORGECODE_AGENT_RE.fullmatch(agent_id):
            raise ValueError(
                "ForgeCode agent_id must contain only letters, digits, "
                "underscore, dot, or dash"
            )
        self.agent_id = agent_id
        self.credentials_path = (
            Path(credentials_path).expanduser() if credentials_path else None
        )
        self.codex_auth_json_path = (
            Path(codex_auth_json_path).expanduser() if codex_auth_json_path else None
        )

    @staticmethod
    @override
    def name() -> str:
        return "forgecode"

    @override
    def get_version_command(self) -> str | None:
        return "forge --version"

    @override
    def parse_version(self, stdout: str) -> str:
        match = _VERSION_RE.search(stdout)
        return match.group(1) if match else stdout.strip()

    async def _installed_forge_satisfies_version(
        self, environment: BaseEnvironment
    ) -> bool:
        result = await environment.exec(command="forge --version")
        if result.return_code != 0:
            return False
        if self._version is None:
            return True
        return self.parse_version(result.stdout or "") == self._version.lstrip("v")

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        if await self._installed_forge_satisfies_version(environment):
            self.logger.debug("ForgeCode is already available")
            return

        requested = self._version or "latest"
        if not re.fullmatch(r"(?:latest|v?[0-9A-Za-z][0-9A-Za-z._-]*)", requested):
            raise ValueError(f"Invalid ForgeCode version: {requested!r}")
        tag = (
            requested
            if requested == "latest" or requested.startswith("v")
            else f"v{requested}"
        )
        release_path = (
            "latest/download" if tag == "latest" else f"download/{shlex.quote(tag)}"
        )

        await self.exec_as_root(
            environment,
            command=(
                "set -euo pipefail; "
                "if command -v apt-get >/dev/null 2>&1; then "
                "apt-get update && apt-get install -y --no-install-recommends "
                "ca-certificates curl; "
                "elif command -v apk >/dev/null 2>&1; then "
                "apk add --no-cache ca-certificates curl; "
                "elif command -v yum >/dev/null 2>&1; then "
                "yum install -y ca-certificates curl; "
                "fi; "
                'case "$(uname -m)" in '
                "x86_64|amd64) forge_arch=x86_64 ;; "
                "aarch64|arm64) forge_arch=aarch64 ;; "
                '*) echo "Unsupported ForgeCode architecture: '
                '$(uname -m)" >&2; exit 1 ;; '
                "esac; "
                "curl -fsSL --retry 3 "
                f'"https://github.com/tailcallhq/forgecode/releases/{release_path}/'
                'forge-${forge_arch}-unknown-linux-gnu" '
                "-o /usr/local/bin/forge; "
                "chmod 0755 /usr/local/bin/forge; "
                "forge --version"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )

    def _split_model(self) -> tuple[str, str]:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError(
                "ForgeCode model must be in provider/model format, "
                "for example codex/gpt-5.6-sol or anthropic/claude-sonnet-4-6"
            )
        provider, model = self.model_name.split("/", 1)
        provider = provider.strip().lower().replace("-", "_")
        model = model.strip()
        if not provider or not model:
            raise ValueError("ForgeCode provider and model must both be non-empty")
        return provider, model

    @staticmethod
    def _jwt_claims(token: str) -> dict[str, Any]:
        parts = token.split(".")
        if len(parts) != 3:
            return {}
        try:
            payload = parts[1] + "=" * (-len(parts[1]) % 4)
            decoded = base64.urlsafe_b64decode(payload)
            claims = json.loads(decoded)
        except (ValueError, json.JSONDecodeError):
            return {}
        return claims if isinstance(claims, dict) else {}

    @classmethod
    def _forge_credentials_from_codex_auth(cls, path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            raise ValueError(f"Codex auth file was not found: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        tokens = data.get("tokens")
        if not isinstance(tokens, dict):
            raise ValueError(f"Codex auth file has no tokens object: {path}")

        access_token = tokens.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise ValueError(f"Codex auth file has no access token: {path}")

        claims = cls._jwt_claims(access_token)
        expires_at_raw = claims.get("exp")
        if not isinstance(expires_at_raw, (int, float)):
            raise ValueError("Codex access token has no usable expiry timestamp")
        expires_at = (
            datetime.fromtimestamp(expires_at_raw, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

        refresh_token = tokens.get("refresh_token")
        if not isinstance(refresh_token, str) or not refresh_token:
            refresh_token = None

        account_id = tokens.get("account_id")
        if not isinstance(account_id, str) or not account_id:
            nested = claims.get("https://api.openai.com/auth")
            account_id = (
                nested.get("chatgpt_account_id") if isinstance(nested, dict) else None
            )

        credential: dict[str, Any] = {
            "id": "codex",
            "auth_details": {
                "o_auth": {
                    "tokens": {
                        "access_token": access_token,
                        "refresh_token": refresh_token,
                        "expires_at": expires_at,
                    },
                    "config": dict(_CODEX_OAUTH_CONFIG),
                }
            },
            "url_params": {},
        }
        if isinstance(account_id, str) and account_id:
            credential["url_params"]["chatgpt_account_id"] = account_id
        return [credential]

    def _resolve_credentials_file(
        self, provider: str
    ) -> tuple[Path | None, Path | None]:
        if self.credentials_path is not None:
            path = self.credentials_path.resolve()
            if not path.is_file():
                raise ValueError(f"ForgeCode credentials file was not found: {path}")
            return path, None

        if provider != "codex":
            return None, None

        path = self.codex_auth_json_path
        if path is None:
            raw_path = self._get_env("CODEX_AUTH_JSON_PATH")
            path = Path(raw_path).expanduser() if raw_path else None
        if path is None:
            return None, None

        credentials = self._forge_credentials_from_codex_auth(path.resolve())
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="harbor-forgecode-",
            suffix=".json",
            delete=False,
        )
        temp_path = Path(handle.name)
        try:
            json.dump(credentials, handle)
            handle.write("\n")
        finally:
            handle.close()
        temp_path.chmod(0o600)
        return temp_path, temp_path

    def _provider_environment(self, provider: str) -> dict[str, str]:
        env: dict[str, str] = {}
        for key in _PROVIDER_ENV_VARS.get(provider, ()):
            value = self._get_env(key)
            if value:
                env[key] = value
        return env

    def _expected_solution_path(self, instruction: str) -> str | None:
        matches = _SOLUTION_RE.findall(instruction)
        if not matches:
            return None
        return f"/root/solution_{matches[-1]}.py"

    @with_prompt_template
    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        provider, model = self._split_model()
        remote_config = self._REMOTE_CONFIG_DIR.as_posix()
        remote_credentials = (self._REMOTE_CONFIG_DIR / ".credentials.json").as_posix()
        output_path = (EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME).as_posix()
        preserved_path = (EnvironmentPaths.agent_dir / "forgecode").as_posix()
        expected_solution = self._expected_solution_path(instruction)

        env = self._provider_environment(provider)
        env.update(self.resolve_env_vars())
        env.update(
            {
                "CI": "true",
                "FORGE_AUTO_INSTALL_VSCODE_EXTENSION": "false",
                "FORGE_CONFIG": remote_config,
                "FORGE_RESTRICTED": "false",
                "FORGE_SESSION__MODEL_ID": model,
                "FORGE_SESSION__PROVIDER_ID": provider,
                "FORGE_TOOL_SUPPORTED": "true",
                "FORGE_UPDATES__AUTO_UPDATE": "false",
                "NO_COLOR": "1",
                "SHELL": "/bin/bash",
                "TERM": "dumb",
            }
        )

        local_credentials: Path | None = None
        temp_credentials: Path | None = None
        try:
            local_credentials, temp_credentials = self._resolve_credentials_file(
                provider
            )
            await self.exec_as_agent(
                environment,
                command=(
                    f"mkdir -p {shlex.quote(remote_config)} "
                    f"{shlex.quote(EnvironmentPaths.agent_dir.as_posix())}"
                ),
                env=env,
            )
            if local_credentials is not None:
                await environment.upload_file(local_credentials, remote_credentials)
                await self.exec_as_agent(
                    environment,
                    command=f"chmod 600 {shlex.quote(remote_credentials)}",
                    env=env,
                )

            if self.skills_dir:
                await self.exec_as_agent(
                    environment,
                    command=(
                        f"mkdir -p {shlex.quote(remote_config)}/skills && "
                        f"cp -R {shlex.quote(self.skills_dir)}/* "
                        f"{shlex.quote(remote_config)}/skills/ 2>/dev/null || true"
                    ),
                    env=env,
                )

            command = (
                "forge "
                f"--agent {shlex.quote(self.agent_id)} "
                "-C /root "
                f"-p {shlex.quote(instruction)} "
                f"2>&1 </dev/null | stdbuf -oL tee {shlex.quote(output_path)}"
            )
            if expected_solution is not None:
                command += f"; test -s {shlex.quote(expected_solution)}"
            await self.exec_as_agent(environment, command=command, env=env)
        finally:
            if temp_credentials is not None:
                try:
                    temp_credentials.unlink()
                except OSError:
                    pass

            try:
                await self.exec_as_agent(
                    environment,
                    command=(
                        f"rm -rf {shlex.quote(preserved_path)}; "
                        f"if [ -d {shlex.quote(remote_config)} ]; then "
                        f"cp -R {shlex.quote(remote_config)} "
                        f"{shlex.quote(preserved_path)}; "
                        f"rm -f {shlex.quote(preserved_path)}/.credentials.json; "
                        "fi; "
                        f"rm -rf {shlex.quote(remote_config)}"
                    ),
                    env=env,
                )
            except Exception:
                pass
