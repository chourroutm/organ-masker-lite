"""Masking pipeline: preflight -> validate -> sweep -> combine -> write."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..backends.base import VideoSegmenterBackend
from ..backends.registry import get_backend
from ..config import RunConfig
from ..io.reader import OmeZarrReader, ReaderError
from ..io.validate import validate_ome_zarr
from ..io.writer import write_mask
from ..prompts.model import PromptSet
from .combine import VoteAccumulator
from .sweep import run_sweep


class PipelineError(RuntimeError):
    """Raised when a masking run cannot proceed."""


def estimate_intermediate_bytes(level_shape: tuple[int, ...]) -> int:
    """Estimate intermediate on-disk bytes: RGB frame stack + uint16 vote accumulator."""
    voxels = int(np.prod(level_shape))
    return voxels * 3 + voxels * 2


def preflight_disk(required_bytes: int, path: str | Path) -> None:
    """Fail clearly if the working directory lacks room for intermediates (FR-016)."""
    free = shutil.disk_usage(str(path)).free
    if required_bytes > free:
        raise PipelineError(
            f"insufficient disk for intermediates: need ~{required_bytes} bytes, "
            f"{free} available at {path}"
        )


def run_masking(
    input_path: str | Path,
    output_path: str | Path,
    prompts: PromptSet,
    config: RunConfig,
    *,
    reader: OmeZarrReader | None = None,
    backend: VideoSegmenterBackend | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Run an end-to-end masking pipeline and write an OME-Zarr v0.5 mask.

    ``reader``/``backend`` may be injected (tests); otherwise they are constructed from
    ``input_path`` and ``config``.
    """

    def report(msg: str) -> None:
        if progress is not None:
            progress(msg)

    report("validating input")
    validate_ome_zarr(input_path)

    reader = reader or OmeZarrReader(input_path)
    level = reader.resolve_level(config.level)

    axis_name = config.axes[0]
    if axis_name not in reader.axes:
        raise PipelineError(f"axis '{axis_name}' not in input axes {reader.axes}")
    axis_index = reader.axes.index(axis_name)
    level_shape = reader.level_shape(level)

    prompts.validate(level_shape, axis_index)

    report(f"reading level {level} {level_shape}")
    volume = reader.read_level(level)

    if backend is None:
        try:
            backend = get_backend(config.backend, config=config)
        except Exception as exc:  # noqa: BLE001
            raise PipelineError(str(exc)) from exc

    with tempfile.TemporaryDirectory(prefix="organ_masker_") as workdir:
        preflight_disk(estimate_intermediate_bytes(level_shape), workdir)
        accumulator = VoteAccumulator(level_shape, workdir)
        report(f"sweeping axis '{axis_name}'")
        mask = run_sweep(volume, axis_index, prompts, backend, workdir, config.direction)
        accumulator.add(mask)
        consensus = accumulator.result(config.combine_rule)

        record = {
            "input": str(input_path),
            "level": level,
            "config": config.to_record(),
            "prompts": prompts.to_record(),
        }
        report("writing output")
        write_mask(output_path, consensus, level, reader, record, overwrite=config.overwrite)
    return Path(output_path)


__all__ = [
    "run_masking",
    "PipelineError",
    "estimate_intermediate_bytes",
    "preflight_disk",
    "ReaderError",
]
