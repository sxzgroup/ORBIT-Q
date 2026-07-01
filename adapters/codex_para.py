from __future__ import annotations

import asyncio
import re
import shlex
import tempfile
import tomllib
from pathlib import Path
from typing import override

from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.codex import Codex
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

_PROFILE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class CodexPara(Codex):
    """Harbor Codex adapter that can run Codex CLI with an optional profile."""

    def __init__(
        self,
        *args,
        profile: str | None = None,
        profile_config_path: str | None = None,
        force_auth_json: bool = False,
        install_retries: int = 3,
        install_retry_delay_sec: float = 5.0,
        model_catalog_path: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if profile == "":
            profile = None
        if profile is not None and (
            profile.startswith("-") or not _PROFILE_RE.fullmatch(profile)
        ):
            raise ValueError(
                "profile must contain only letters, digits, underscore, dot, or dash"
            )

        self.profile = profile
        self.profile_config_path = (
            Path(profile_config_path).expanduser() if profile_config_path else None
        )
        self.force_auth_json = force_auth_json
        self.install_retries = max(1, int(install_retries))
        self.install_retry_delay_sec = max(0.0, float(install_retry_delay_sec))
        self.model_catalog_path = (
            Path(model_catalog_path).expanduser() if model_catalog_path else None
        )

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        for attempt in range(1, self.install_retries + 1):
            try:
                await self._install_once(environment)
                return
            except Exception:
                if attempt >= self.install_retries:
                    raise
                self.logger.exception(
                    "Codex install attempt %s/%s failed; retrying",
                    attempt,
                    self.install_retries,
                )
                await asyncio.sleep(self.install_retry_delay_sec * attempt)

    async def _install_once(self, environment: BaseEnvironment) -> None:
        if await self._installed_codex_satisfies_version(environment):
            self.logger.debug("Codex is already available at the requested version")
            return

        await self.exec_as_root(
            environment,
            command=(
                "if ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]; then"
                "  apk add --no-cache bash nodejs npm ripgrep;"
                " elif command -v apt-get &>/dev/null; then"
                "  apt-get update && apt-get install -y --no-install-recommends "
                "ca-certificates nodejs npm ripgrep;"
                " elif command -v yum &>/dev/null; then"
                "  yum install -y ca-certificates nodejs npm ripgrep;"
                " else"
                '  echo "Warning: No known package manager found, assuming npm is available" >&2;'
                " fi"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )

        version_spec = f"@{self._version}" if self._version else "@latest"
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "npm config set registry https://registry.npmmirror.com; "
                f"npm install -g @openai/codex{version_spec}; "
                "codex --version"
            ),
        )

        await self.exec_as_root(
            environment,
            command=(
                "for bin in node npm npx codex; do"
                '  BIN_PATH="$(which "$bin" 2>/dev/null || true)";'
                '  if [ -n "$BIN_PATH" ] && [ "$BIN_PATH" != "/usr/local/bin/$bin" ]; then'
                '    ln -sf "$BIN_PATH" "/usr/local/bin/$bin";'
                "  fi;"
                " done"
            ),
        )

    def _resolve_profile_config_path(self) -> Path | None:
        explicit = self.profile_config_path
        if explicit is None:
            env_path = self._get_env("CODEX_PROFILE_CONFIG_PATH")
            explicit = Path(env_path).expanduser() if env_path else None

        if explicit is None and self.profile is None:
            return None

        path = explicit or (Path.home() / ".codex" / f"{self.profile}.config.toml")
        if not path.is_file():
            raise ValueError(
                f"Codex config for profile '{self.profile or 'default'}' was not found: {path}"
            )
        return path

    def _profile_config(self) -> dict:
        path = self._resolve_profile_config_path()
        if path is None:
            return {}
        return tomllib.loads(path.read_text())

    def _profile_env_key(self) -> str | None:
        config = self._profile_config()
        provider_name = config.get("model_provider")
        providers = config.get("model_providers")
        if not isinstance(provider_name, str) or not isinstance(providers, dict):
            return None
        provider = providers.get(provider_name)
        if not isinstance(provider, dict):
            return None
        env_key = provider.get("env_key")
        return env_key if isinstance(env_key, str) and env_key else None

    def _resolve_model_catalog_path(self) -> Path | None:
        if self.model_catalog_path is not None:
            path = self.model_catalog_path
        else:
            raw_path = self._profile_config().get("model_catalog_json")
            if not isinstance(raw_path, str) or not raw_path:
                return None
            path = Path(raw_path).expanduser()

        if not path.is_file():
            raise ValueError(f"Codex model catalog was not found: {path}")
        return path

    @override
    def _resolve_auth_json_path(self) -> Path | None:
        auth_json_path = super()._resolve_auth_json_path()
        if auth_json_path is not None or not self.force_auth_json:
            return auth_json_path

        default = Path.home() / ".codex" / "auth.json"
        if not default.is_file():
            raise ValueError(
                "force_auth_json is enabled, but the default Codex auth file does not exist. "
                "Pass --ak force_auth_json=false to use OPENAI_API_KEY auth instead."
            )
        return default

    async def _upload_profile_config(
        self,
        environment: BaseEnvironment,
    ) -> None:
        local_profile_config = self._resolve_profile_config_path()
        if local_profile_config is None:
            return

        remote_config_name = (
            f"{self.profile}.config.toml" if self.profile else "config.toml"
        )
        remote_profile_config = (self._REMOTE_CODEX_HOME / remote_config_name).as_posix()

        await environment.upload_file(local_profile_config, remote_profile_config)
        if environment.default_user is not None:
            await self.exec_as_root(
                environment,
                command=(
                    f"chown {shlex.quote(str(environment.default_user))} "
                    f"{shlex.quote(remote_profile_config)}"
                ),
            )

        local_model_catalog = self._resolve_model_catalog_path()
        if local_model_catalog is None:
            return

        remote_model_catalog = (self._REMOTE_CODEX_HOME / "models.json").as_posix()
        remote_home_catalog = "$HOME/.codex/models.json"
        await environment.upload_file(local_model_catalog, remote_model_catalog)
        await self.exec_as_agent(
            environment,
            command=(
                'mkdir -p "$HOME/.codex" && '
                f"cp {shlex.quote(remote_model_catalog)} {remote_home_catalog}"
            ),
            env={"CODEX_HOME": self._REMOTE_CODEX_HOME.as_posix()},
        )
        if environment.default_user is not None:
            await self.exec_as_root(
                environment,
                command=(
                    f"chown {shlex.quote(str(environment.default_user))} "
                    f"{shlex.quote(remote_model_catalog)} "
                    '"$HOME/.codex/models.json"'
                ),
            )

    def _build_codex_exec_command(
        self,
        *,
        model: str,
        cli_flags_arg: str,
        escaped_instruction: str,
    ) -> str:
        output_path = (EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME).as_posix()
        profile_arg = (
            f"--profile {shlex.quote(self.profile)} " if self.profile else ""
        )

        return (
            "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
            f"if [ -s {shlex.quote((self._REMOTE_CODEX_SECRETS_DIR / 'profile_env.sh').as_posix())} ]; then "
            f". {shlex.quote((self._REMOTE_CODEX_SECRETS_DIR / 'profile_env.sh').as_posix())}; "
            "fi; "
            "codex exec "
            f"{profile_arg}"
            "--dangerously-bypass-approvals-and-sandbox "
            "--skip-git-repo-check "
            f"--model {shlex.quote(model)} "
            "--json "
            "--enable unified_exec "
            f"{cli_flags_arg}"
            "-- "
            f"{escaped_instruction} "
            f"2>&1 </dev/null | tee {shlex.quote(output_path)}"
        )

    async def _upload_profile_env(
        self,
        environment: BaseEnvironment,
        remote_secrets_dir: str,
    ) -> None:
        profile_env_key = self._profile_env_key()
        if not profile_env_key:
            return

        profile_env_value = self._get_env(profile_env_key)
        if not profile_env_value:
            return

        remote_env_path = (self._REMOTE_CODEX_SECRETS_DIR / "profile_env.sh").as_posix()

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", delete=False) as handle:
                temp_path = Path(handle.name)
                handle.write(
                    f"export {profile_env_key}={shlex.quote(profile_env_value)}\n"
                )
            temp_path.chmod(0o600)

            await environment.upload_file(temp_path, remote_env_path)
            await self.exec_as_agent(
                environment,
                command=f"chmod 600 {shlex.quote(remote_env_path)}",
                env={"CODEX_HOME": self._REMOTE_CODEX_HOME.as_posix()},
            )
            if environment.default_user is not None:
                await self.exec_as_root(
                    environment,
                    command=(
                        f"chown {shlex.quote(str(environment.default_user))} "
                        f"{shlex.quote(remote_env_path)}"
                    ),
                )
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    @with_prompt_template
    @override
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        escaped_instruction = shlex.quote(instruction)

        if not self.model_name:
            raise ValueError("Model name is required")

        model = self.model_name.split("/")[-1]

        cli_flags = self.build_cli_flags()
        cli_flags_arg = (cli_flags + " ") if cli_flags else ""

        auth_json_path = self._resolve_auth_json_path()

        remote_codex_home = self._REMOTE_CODEX_HOME.as_posix()
        remote_secrets_dir = self._REMOTE_CODEX_SECRETS_DIR.as_posix()
        remote_auth_path = (self._REMOTE_CODEX_SECRETS_DIR / "auth.json").as_posix()

        env: dict[str, str] = {
            "CODEX_HOME": remote_codex_home,
        }

        await self.exec_as_agent(
            environment,
            command=(
                f'mkdir -p "$CODEX_HOME" {shlex.quote(remote_secrets_dir)} '
                f"{shlex.quote(EnvironmentPaths.agent_dir.as_posix())}"
            ),
            env=env,
        )

        await self._upload_profile_config(environment)
        await self._upload_profile_env(environment, remote_secrets_dir)

        if auth_json_path:
            self.logger.debug("Codex auth: using auth.json")
            await environment.upload_file(auth_json_path, remote_auth_path)
            if environment.default_user is not None:
                await self.exec_as_root(
                    environment,
                    command=f"chown {environment.default_user} {remote_auth_path}",
                )
            setup_command = (
                f'ln -sf {shlex.quote(remote_auth_path)} "$CODEX_HOME/auth.json"\n'
            )
        else:
            self.logger.debug("Codex auth: using OPENAI_API_KEY")
            env["OPENAI_API_KEY"] = self._get_env("OPENAI_API_KEY") or ""
            setup_command = (
                f"cat >{shlex.quote(remote_auth_path)} <<EOF\n"
                '{\n  "OPENAI_API_KEY": "${OPENAI_API_KEY}"\n}\nEOF\n'
                f"ln -sf {shlex.quote(remote_auth_path)} "
                '"$CODEX_HOME/auth.json"\n'
            )

        if openai_base_url := self._get_env("OPENAI_BASE_URL"):
            env["OPENAI_BASE_URL"] = openai_base_url

        config_toml_block = ""
        if openai_base_url:
            config_toml_block = (
                '\ncat >>"$CODEX_HOME/config.toml" <<TOML\n'
                'openai_base_url = "${OPENAI_BASE_URL}"\n'
                "TOML"
            )

        setup_command += config_toml_block

        skills_command = self._build_register_skills_command()
        if skills_command:
            setup_command += f"\n{skills_command}"

        mcp_command = self._build_register_mcp_servers_command()
        if mcp_command:
            setup_command += f"\n{mcp_command}"

        if setup_command.strip():
            await self.exec_as_agent(
                environment,
                command=setup_command,
                env=env,
            )

        try:
            await self.exec_as_agent(
                environment,
                command=self._build_codex_exec_command(
                    model=model,
                    cli_flags_arg=cli_flags_arg,
                    escaped_instruction=escaped_instruction,
                ),
                env=env,
            )
        finally:
            try:
                sessions_dir = (EnvironmentPaths.agent_dir / "sessions").as_posix()
                await self.exec_as_agent(
                    environment,
                    command=(
                        f"mkdir -p {EnvironmentPaths.agent_dir.as_posix()}\n"
                        'if [ -d "$CODEX_HOME/sessions" ]; then\n'
                        f"  rm -rf {sessions_dir}\n"
                        f'  cp -R "$CODEX_HOME/sessions" {sessions_dir}\n'
                        "fi"
                    ),
                    env=env,
                )
            except Exception:
                pass

            try:
                await self.exec_as_agent(
                    environment,
                    command=f'rm -rf {shlex.quote(remote_secrets_dir)} "$CODEX_HOME"',
                    env=env,
                )
            except Exception:
                pass
