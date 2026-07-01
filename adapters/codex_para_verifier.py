from __future__ import annotations

import shlex
from pathlib import Path
from typing import override

from harbor.models.verifier.result import VerifierResult
from harbor.verifier.verifier import Verifier

from adapters.codex_para import CodexPara


class CodexParaVerifier(Verifier):
    """Verifier wrapper that prepares Codex CLI for the source audit.

    The task's ordinary tests still own scoring. This class only installs Codex
    in the verifier container and injects optional profile/auth runtime needed
    by tests that run an LLM source audit.
    """

    def __init__(
        self,
        *args,
        profile: str | None = None,
        audit_model: str = "gpt-5",
        profile_config_path: str | None = None,
        force_auth_json: bool = False,
        install_retries: int = 3,
        install_retry_delay_sec: float = 5.0,
        model_catalog_path: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.audit_model = audit_model
        self._codex = CodexPara(
            logs_dir=self.trial_paths.verifier_dir,
            model_name=audit_model,
            profile=profile,
            profile_config_path=profile_config_path,
            force_auth_json=force_auth_json,
            install_retries=install_retries,
            install_retry_delay_sec=install_retry_delay_sec,
            model_catalog_path=model_catalog_path,
            logger=self.logger,
        )

    async def _setup_codex_runtime(self) -> None:
        await self._codex.install(self.environment)

        remote_codex_home = self._codex._REMOTE_CODEX_HOME.as_posix()
        remote_secrets_dir = self._codex._REMOTE_CODEX_SECRETS_DIR.as_posix()
        remote_auth_path = (
            self._codex._REMOTE_CODEX_SECRETS_DIR / "auth.json"
        ).as_posix()

        env: dict[str, str] = {"CODEX_HOME": remote_codex_home}
        await self._codex.exec_as_agent(
            self.environment,
            command=f'mkdir -p "$CODEX_HOME" {shlex.quote(remote_secrets_dir)}',
            env=env,
        )

        await self._codex._upload_profile_config(self.environment)
        await self._codex._upload_profile_env(self.environment, remote_secrets_dir)

        auth_json_path = self._codex._resolve_auth_json_path()
        if auth_json_path:
            await self.environment.upload_file(Path(auth_json_path), remote_auth_path)
            if self.environment.default_user is not None:
                await self._codex.exec_as_root(
                    self.environment,
                    command=(
                        f"chown {shlex.quote(str(self.environment.default_user))} "
                        f"{shlex.quote(remote_auth_path)}"
                    ),
                )
            setup_command = (
                f'ln -sf {shlex.quote(remote_auth_path)} "$CODEX_HOME/auth.json"'
            )
        else:
            env["OPENAI_API_KEY"] = self._codex._get_env("OPENAI_API_KEY") or ""
            setup_command = (
                f"cat >{shlex.quote(remote_auth_path)} <<EOF\n"
                '{\n  "OPENAI_API_KEY": "${OPENAI_API_KEY}"\n}\nEOF\n'
                f"ln -sf {shlex.quote(remote_auth_path)} "
                '"$CODEX_HOME/auth.json"'
            )

        if openai_base_url := self._codex._get_env("OPENAI_BASE_URL"):
            env["OPENAI_BASE_URL"] = openai_base_url
            setup_command += (
                '\ncat >>"$CODEX_HOME/config.toml" <<TOML\n'
                'openai_base_url = "${OPENAI_BASE_URL}"\n'
                "TOML"
            )

        await self._codex.exec_as_agent(
            self.environment,
            command=setup_command,
            env=env,
        )

        self.override_env.update(
            {
                "CODEX_HOME": remote_codex_home,
                "CODEX_AUDIT_MODEL": self.audit_model.split("/")[-1],
                "CODEX_PROFILE_ENV_FILE": (
                    self._codex._REMOTE_CODEX_SECRETS_DIR / "profile_env.sh"
                ).as_posix(),
            }
        )
        if self._codex.profile:
            self.override_env["CODEX_PROFILE"] = self._codex.profile

    @override
    async def verify(self) -> VerifierResult:
        await self._setup_codex_runtime()
        try:
            return await super().verify()
        finally:
            try:
                await self._codex.exec_as_agent(
                    self.environment,
                    command=(
                        f"rm -rf {shlex.quote(self._codex._REMOTE_CODEX_SECRETS_DIR.as_posix())} "
                        f"{shlex.quote(self._codex._REMOTE_CODEX_HOME.as_posix())}"
                    ),
                    env={"CODEX_HOME": self._codex._REMOTE_CODEX_HOME.as_posix()},
                )
            except Exception:
                pass
