"""Segmentation backend protocol.

A backend propagates a segmentation across an ordered stack of 2D frames (one axis sweep),
guided by SAM2-convention prompts, and returns a per-frame binary mask.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..prompts.model import PromptSet


@runtime_checkable
class VideoSegmenterBackend(Protocol):
    """Segment a stack of frames given prompts placed on specific frames."""

    name: str

    def segment_video(self, frames: np.ndarray, prompts: PromptSet) -> np.ndarray:
        """Return a ``(T, H, W)`` boolean mask for ``frames`` of shape ``(T, H, W, 3)`` uint8.

        Prompt ``frame_index`` indexes the first (T) dimension; ``point_coords`` are ``(x, y)``
        with ``x`` along W and ``y`` along H.
        """
        ...
