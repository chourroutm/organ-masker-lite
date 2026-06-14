"""SAM2-compatible landmark prompts.

Mirrors the SAM2 prompt-encoder convention: point coordinates are ``(x, y)`` in slice pixel
space, labels are ``1`` (positive/include) or ``0`` (negative/exclude); an optional box is
``[x_min, y_min, x_max, y_max]``. Box-corner labels (2/3) and padding (-1) are produced
internally by the backend and are never supplied here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class PromptError(ValueError):
    """Raised when a prompt or prompt set is invalid."""


@dataclass
class Prompt:
    """Landmark prompts placed on one slice/frame for one object."""

    frame_index: int
    point_coords: np.ndarray  # (N, 2) float, (x, y)
    point_labels: np.ndarray  # (N,) int in {0, 1}
    box: np.ndarray | None = None  # (4,) float [x0, y0, x1, y1]
    obj_id: int = 0

    def __post_init__(self) -> None:
        self.point_coords = np.asarray(self.point_coords, dtype=float).reshape(-1, 2)
        self.point_labels = np.asarray(self.point_labels, dtype=int).reshape(-1)
        if self.point_coords.shape[0] != self.point_labels.shape[0]:
            raise PromptError("point_coords and point_labels must have the same length")
        if self.point_labels.size and not np.isin(self.point_labels, (0, 1)).all():
            raise PromptError("point_labels must be 0 (exclude) or 1 (include)")
        if self.box is not None:
            self.box = np.asarray(self.box, dtype=float).reshape(4)

    @property
    def positive_coords(self) -> np.ndarray:
        return self.point_coords[self.point_labels == 1]

    @property
    def negative_coords(self) -> np.ndarray:
        return self.point_coords[self.point_labels == 0]


@dataclass
class PromptSet:
    """A collection of prompts for one run (single target structure, FR-021)."""

    prompts: list[Prompt] = field(default_factory=list)

    def has_positive(self) -> bool:
        return any(p.point_labels.tolist().count(1) or p.box is not None for p in self.prompts)

    def validate(self, level_shape: tuple[int, ...], axis_index: int) -> None:
        """Validate prompts against the selected level (FR-015).

        ``level_shape`` is the (z, y, x) shape of the run level; ``axis_index`` is the sweep
        axis. Coordinates are checked against the in-plane dimensions of that axis.
        """
        if not self.has_positive():
            raise PromptError("at least one positive point or a box is required")
        plane = [d for i, d in enumerate(level_shape) if i != axis_index]
        height, width = plane[0], plane[1]
        n_frames = level_shape[axis_index]
        for p in self.prompts:
            if not (0 <= p.frame_index < n_frames):
                raise PromptError(
                    f"frame_index {p.frame_index} out of range [0, {n_frames}) for the sweep axis"
                )
            for x, y in p.point_coords:
                if not (0 <= x < width and 0 <= y < height):
                    raise PromptError(
                        f"point ({x}, {y}) out of bounds for slice of size (h={height}, w={width})"
                    )
            if p.box is not None:
                x0, y0, x1, y1 = p.box
                if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                    raise PromptError(f"box {p.box.tolist()} out of bounds or malformed")

    def to_record(self) -> list[dict]:
        out = []
        for p in self.prompts:
            out.append(
                {
                    "obj_id": p.obj_id,
                    "frame_index": p.frame_index,
                    "points": [
                        {"xy": [float(x), float(y)], "label": int(lbl)}
                        for (x, y), lbl in zip(p.point_coords, p.point_labels, strict=True)
                    ],
                    "box": None if p.box is None else [float(v) for v in p.box],
                }
            )
        return out


def load_prompt_file(path: str | Path) -> PromptSet:
    """Load a prompt file in the documented JSON format (see contracts/cli.md)."""
    data = json.loads(Path(path).read_text())
    prompts: list[Prompt] = []
    for obj in data.get("objects", []):
        obj_id = int(obj.get("obj_id", 0))
        by_frame: dict[int, dict] = {}
        for pt in obj.get("points", []):
            f = int(pt["frame"])
            entry = by_frame.setdefault(f, {"coords": [], "labels": []})
            entry["coords"].append(pt["xy"])
            entry["labels"].append(int(pt["label"]))
        box = obj.get("box")
        box_frame = int(box["frame"]) if box else None
        frames = set(by_frame) | ({box_frame} if box_frame is not None else set())
        for f in sorted(frames):
            entry = by_frame.get(f, {"coords": [], "labels": []})
            prompts.append(
                Prompt(
                    frame_index=f,
                    point_coords=entry["coords"] if entry["coords"] else np.empty((0, 2)),
                    point_labels=entry["labels"] if entry["labels"] else np.empty((0,), int),
                    box=np.asarray(box["xyxy"], float) if box and box_frame == f else None,
                    obj_id=obj_id,
                )
            )
    if not prompts:
        raise PromptError("prompt file contains no prompts")
    return PromptSet(prompts=prompts)
