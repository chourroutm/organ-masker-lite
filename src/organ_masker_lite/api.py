"""SAM-like programmatic API: a thin facade over the shared engine (FR-010, SC-004).

Mirrors the structure of the SAM/SAM2 predictor so SAM users adopt it with minimal learning:

    from organ_masker_lite import OrganMaskPredictor

    predictor = OrganMaskPredictor(backend="sam2")
    predictor.set_volume("input.ome.zarr", level=3)          # mirrors SAM set_image
    predictor.add_points(frame_index=120, point_coords=[[x, y]], point_labels=[1])
    mask = predictor.predict(axes=["z"], direction="forward")
    mask.save("output.ome.zarr")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import COARSEST_LEVEL, PostProcessConfig, RunConfig
from .engine.pipeline import compute_mask
from .io.reader import OmeZarrReader
from .io.validate import validate_ome_zarr
from .io.writer import write_mask
from .prompts.model import Prompt, PromptError, PromptSet


class MaskResult:
    """A computed consensus mask at the run level, with the data needed to write it (FR-014)."""

    def __init__(self, array: np.ndarray, level: int, reader: OmeZarrReader, record: dict):
        self.array = array
        self._level = level
        self._reader = reader
        self._record = record

    def save(self, store_path: str | Path, overwrite: bool = False) -> Path:
        """Write OME-Zarr v0.5 with the same levels as the input, embedding the run record.

        Refuses to overwrite an existing store unless ``overwrite=True`` (FR-013).
        """
        return write_mask(
            store_path, self.array, self._level, self._reader, self._record, overwrite=overwrite
        )


class OrganMaskPredictor:
    """SAM-like predictor facade producing OME-Zarr v0.5 masks from landmark prompts."""

    def __init__(
        self,
        backend: str = "sam2",
        model_dir: str | Path | None = None,
        allow_download: bool = True,
    ):
        self._backend = backend
        self._model_dir = model_dir
        self._allow_download = allow_download
        self._store_path: str | None = None
        self._reader: OmeZarrReader | None = None
        self._level: int = COARSEST_LEVEL
        self._prompts: list[Prompt] = []

    def set_volume(self, store_path: str | Path, level: int = COARSEST_LEVEL) -> OrganMaskPredictor:
        """Validate the store, select the level, and prepare lazy access. Mirrors SAM ``set_image``.

        Setting a new volume clears any previously added prompts (as SAM resets on a new image).
        """
        validate_ome_zarr(store_path)  # FR-001
        reader = OmeZarrReader(store_path)
        self._level = reader.resolve_level(level)  # validates the level (FR-003)
        self._reader = reader
        self._store_path = str(store_path)
        self._prompts = []
        return self

    def add_points(
        self,
        frame_index: int,
        point_coords,
        point_labels,
        obj_id: int = 0,
    ) -> OrganMaskPredictor:
        """Add point prompts; argument names/shapes match SAM2's ``add_new_points_or_box``.

        ``point_coords`` is ``(N, 2)`` of ``(x, y)``; ``point_labels`` is ``(N,)`` with ``1``
        positive and ``0`` negative/exclusion (research R4).
        """
        self._require_volume()
        self._prompts.append(
            Prompt(
                frame_index=frame_index,
                point_coords=point_coords,
                point_labels=point_labels,
                obj_id=obj_id,
            )
        )
        return self

    def add_box(self, frame_index: int, box, obj_id: int = 0) -> OrganMaskPredictor:
        """Add a box prompt ``[x_min, y_min, x_max, y_max]`` (SAM2 convention)."""
        self._require_volume()
        self._prompts.append(
            Prompt(
                frame_index=frame_index,
                point_coords=np.empty((0, 2)),
                point_labels=np.empty((0,), dtype=int),
                box=box,
                obj_id=obj_id,
            )
        )
        return self

    def predict(
        self,
        axes: list[str] | None = None,
        direction: str = "forward",
        combine_rule: str = "majority",
        postprocess: PostProcessConfig | None = None,
    ) -> MaskResult:
        """Sweep, combine, and return a :class:`MaskResult` (equivalent to CLI output, SC-005)."""
        self._require_volume()
        kwargs: dict = {
            "backend": self._backend,
            "level": self._level,
            "direction": direction,
            "combine_rule": combine_rule,
            "postprocess": postprocess or PostProcessConfig(),
            "model_dir": Path(self._model_dir) if self._model_dir is not None else None,
            "allow_download": self._allow_download,
        }
        if axes is not None:
            kwargs["axes"] = list(axes)
        config = RunConfig(**kwargs)

        comp = compute_mask(
            self._store_path, PromptSet(list(self._prompts)), config, reader=self._reader
        )
        return MaskResult(comp.consensus, comp.level, comp.reader, comp.record)

    def _require_volume(self) -> None:
        if self._reader is None:
            raise PromptError("call set_volume(...) before adding prompts or predicting")


__all__ = ["OrganMaskPredictor", "MaskResult"]
