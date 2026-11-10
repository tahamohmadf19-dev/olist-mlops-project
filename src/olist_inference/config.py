from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(ValueError):
    """Invalid or incomplete inference configuration."""


@dataclass(frozen=True)
class InferenceConfig:
    preprocessor_path: Path
    model_path: Path
    summary_path: Path
    max_batch_size: int

    @classmethod
    def from_json(cls, filename: str | Path) -> "InferenceConfig":
        config_file = Path(filename).resolve()
        if not config_file.is_file():
            raise ConfigurationError(f"Missing configuration: {config_file}")

        try:
            raw = json.loads(config_file.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ConfigurationError("Unable to read inference configuration") from exc
        if not isinstance(raw, dict):
            raise ConfigurationError("Configuration must be a JSON object")

        project_root = config_file.parent.parent

        def artifact_path(key: str) -> Path:
            value = raw.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"Missing configuration entry: {key}")
            path = Path(value)
            if path.is_absolute():
                raise ConfigurationError(f"Expected project-relative path for {key}")
            resolved = (project_root / path).resolve()
            if not resolved.is_relative_to(project_root):
                raise ConfigurationError(f"Path escapes project root: {key}")
            return resolved

        batch_size = raw.get("max_batch_size", 100)
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or not 1 <= batch_size <= 1000
        ):
            raise ConfigurationError("max_batch_size must be an integer from 1 to 1000")

        return cls(
            preprocessor_path=artifact_path("preprocessor_path"),
            model_path=artifact_path("model_path"),
            summary_path=artifact_path("summary_path"),
            max_batch_size=batch_size,
        )
