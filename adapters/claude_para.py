from __future__ import annotations

import re
from typing import override

from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.installed.claude_code import ClaudeCode
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

_SOLUTION_RE = re.compile(r"\bsolution_(\d+)\.py\b")


class ClaudePara(ClaudeCode):
    """Claude Code adapter that prefers the CLI preinstalled in framework images."""

    def __init__(self, *args, accept_solution_on_error: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.accept_solution_on_error = accept_solution_on_error

    async def _preinstalled_claude_satisfies_version(
        self, environment: BaseEnvironment
    ) -> bool:
        result = await environment.exec(command=self.get_version_command() or "")
        if result.return_code != 0:
            return False
        if self._version is None:
            return True
        installed_version = self.parse_version(result.stdout or "")
        return installed_version == self._version

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        if await self._preinstalled_claude_satisfies_version(environment):
            self.logger.debug("Claude Code is already available")
            return
        await super().install(environment)

    def _expected_solution_path(self, instruction: str) -> str | None:
        matches = _SOLUTION_RE.findall(instruction)
        if not matches:
            return None
        return f"/root/solution_{matches[-1]}.py"

    @override
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        try:
            await super().run(instruction, environment, context)
        except NonZeroAgentExitCodeError:
            if not self.accept_solution_on_error:
                raise

            solution_path = self._expected_solution_path(instruction)
            if solution_path is None:
                raise

            result = await environment.exec(command=f"test -s {solution_path}")
            if result.return_code != 0:
                raise

            self.logger.warning(
                "Claude Code exited nonzero after producing %s; "
                "continuing so the verifier can score the artifact.",
                solution_path,
            )
