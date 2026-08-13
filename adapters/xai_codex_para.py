from __future__ import annotations

import shlex
from pathlib import Path

from harbor.environments.base import BaseEnvironment

from adapters.codex_para import CodexPara


class XAICodexPara(CodexPara):
    """Codex adapter with a narrow xAI Responses compatibility shim.

    Codex 0.146 drops ``status`` and adds ``content: null`` when replaying an
    encrypted reasoning item.  xAI rejects that altered shape.  The local shim
    reconstructs the original xAI item shape.  It also preserves Codex integer
    tool schemas and converts only integral JSON floats in integer arguments
    back to integers, avoiding xAI/Codex router type mismatches.
    """

    def __init__(
        self,
        *args,
        responses_proxy_path: str,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.responses_proxy_path = Path(responses_proxy_path).expanduser()
        if not self.responses_proxy_path.is_file():
            raise ValueError(
                f"xAI Responses proxy was not found: {self.responses_proxy_path}"
            )

    async def _upload_profile_config(self, environment: BaseEnvironment) -> None:
        await super()._upload_profile_config(environment)
        remote_proxy = (self._REMOTE_CODEX_HOME / "xai_responses_proxy.py").as_posix()
        await environment.upload_file(self.responses_proxy_path, remote_proxy)
        if environment.default_user is not None:
            await self.exec_as_root(
                environment,
                command=(
                    f"chown {shlex.quote(str(environment.default_user))} "
                    f"{shlex.quote(remote_proxy)}"
                ),
            )

    def _build_codex_exec_command(
        self,
        *,
        model: str,
        cli_flags_arg: str,
        escaped_instruction: str,
    ) -> str:
        base_command = super()._build_codex_exec_command(
            model=model,
            cli_flags_arg=cli_flags_arg,
            escaped_instruction=escaped_instruction,
        )
        remote_proxy = (self._REMOTE_CODEX_HOME / "xai_responses_proxy.py").as_posix()
        proxy_log = "/logs/agent/xai-responses-proxy.jsonl"
        return (
            f"python3 {shlex.quote(remote_proxy)} --host 127.0.0.1 --port 8877 "
            "--restore-reasoning-status --normalize-integral-tool-arguments "
            f"> {shlex.quote(proxy_log)} 2>&1 & "
            "XAI_PROXY_PID=$!; "
            "trap 'kill \"$XAI_PROXY_PID\" 2>/dev/null || true' EXIT; "
            f"{base_command}"
        )
