"""Run and post-processing configuration, and model-directory resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL_DIR = "organ_masker_models"
MODEL_DIR_ENV = "ORGAN_MASKER_MODEL_DIR"

DEFAULT_LOG_DIR = "organ_masker_logs"
LOG_DIR_ENV = "ORGAN_MASKER_LOG_DIR"

#: Sentinel meaning "use the coarsest level (highest index)".
COARSEST_LEVEL = -1


@dataclass
class LogConfig:
    """Configuration for per-invocation input logging (feature 002, FR-007)."""

    log_dir: Path | None = None
    level: str = "INFO"
    enabled: bool = True

    def resolved_log_dir(self) -> Path:
        """Resolve the log directory: explicit > env var > default (cwd subdir) (FR-007)."""
        if self.log_dir is not None:
            return Path(self.log_dir)
        env = os.environ.get(LOG_DIR_ENV)
        if env:
            return Path(env)
        return Path.cwd() / DEFAULT_LOG_DIR


@dataclass
class PostProcessConfig:
    """Optional mask post-processing (disabled morphology, hole-filling on by default)."""

    fill_holes: bool = True
    dilation_radius: int = 0
    erosion_radius: int = 0

    def __post_init__(self) -> None:
        if self.dilation_radius < 0 or self.erosion_radius < 0:
            raise ValueError("dilation_radius and erosion_radius must be non-negative")

    @property
    def is_noop(self) -> bool:
        return not self.fill_holes and self.dilation_radius == 0 and self.erosion_radius == 0


@dataclass
class RunConfig:
    """The complete, reproducible parameterization of a masking run (FR-014)."""

    backend: str = "sam2"
    level: int = COARSEST_LEVEL
    axes: list[str] = field(default_factory=lambda: ["z"])
    direction: str = "forward"
    combine_rule: str = "majority"
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)
    model_dir: Path | None = None
    allow_download: bool = True
    overwrite: bool = False

    def __post_init__(self) -> None:
        # ``level`` is either the COARSEST sentinel or a concrete non-negative index (FR-003).
        if self.level != COARSEST_LEVEL and self.level < 0:
            raise ValueError(
                f"invalid level {self.level}; use a non-negative index or "
                f"{COARSEST_LEVEL} for the coarsest (default) level"
            )

    def resolved_model_dir(self) -> Path:
        """Resolve the model directory: explicit > env var > default (cwd subdir) (FR-019)."""
        if self.model_dir is not None:
            return Path(self.model_dir)
        env = os.environ.get(MODEL_DIR_ENV)
        if env:
            return Path(env)
        return Path.cwd() / DEFAULT_MODEL_DIR

    def to_record(self) -> dict:
        """Serialize the configuration for the reproducibility run-record (FR-014)."""
        return {
            "backend": self.backend,
            "level": self.level,
            "axes": list(self.axes),
            "direction": self.direction,
            "combine_rule": self.combine_rule,
            "postprocess": {
                "fill_holes": self.postprocess.fill_holes,
                "dilation_radius": self.postprocess.dilation_radius,
                "erosion_radius": self.postprocess.erosion_radius,
            },
            "model_dir": str(self.resolved_model_dir()),
            "allow_download": self.allow_download,
        }
