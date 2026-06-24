from __future__ import annotations

import os
import re
from pathlib import Path

from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9_.-]+", "-", value.lower()).strip("-")
    if not slug:
        raise ValueError("framework slug cannot be empty")
    return slug


class FrameworkDockerEnvironment(DockerEnvironment):
    """Docker environment that selects the quantum framework image at run time."""

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        *args,
        framework: str | None = None,
        docker_image: str | None = None,
        image_template: str = "challenge-benchmark-quantum-{framework}:py311",
        **kwargs,
    ) -> None:
        selected_framework = framework or os.environ.get(
            "REQUIRED_QUANTUM_FRAMEWORK", "tensorcircuit"
        )
        framework_slug = _slugify(selected_framework)
        selected_image = docker_image or os.environ.get("DOCKER_IMAGE")
        if selected_image is None:
            selected_image = image_template.format(framework=framework_slug)

        task_env_config = task_env_config.model_copy(
            update={"docker_image": selected_image}
        )
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            *args,
            **kwargs,
        )
