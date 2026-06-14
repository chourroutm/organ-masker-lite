"""Single-axis sweep: present a volume as a frame stack and segment it via the backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..backends.base import VideoSegmenterBackend
from ..prompts.model import PromptSet


def run_sweep(
    volume: np.ndarray,
    axis_index: int,
    prompts: PromptSet,
    backend: VideoSegmenterBackend,
    workdir: str | Path,
    direction: str = "forward",
) -> np.ndarray:
    """Sweep ``volume`` along ``axis_index`` and return a 3D boolean mask in volume coordinates.

    The sweep axis is moved to axis 0 to form a ``(T, H, W)`` frame stack; the backend returns a
    ``(T, H, W)`` mask which is moved back. Reverse propagation is added in user story 2.
    """
    from .frames import build_frame_stack

    if direction not in ("forward", "forward_reverse"):
        raise ValueError(f"unknown direction '{direction}'")
    moved = np.moveaxis(volume, axis_index, 0)
    frames = build_frame_stack(moved, workdir)
    mask_axis0 = backend.segment_video(frames, prompts)
    mask_axis0 = np.asarray(mask_axis0, dtype=bool)
    if mask_axis0.shape != moved.shape:
        raise ValueError(
            f"backend returned mask of shape {mask_axis0.shape}, expected {moved.shape}"
        )
    return np.moveaxis(mask_axis0, 0, axis_index)
