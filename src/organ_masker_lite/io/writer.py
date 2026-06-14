"""Write a binary mask as an OME-Zarr v0.5 image matching the input's levels."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import zarr
from ome_zarr_models.v05.axes import Axis
from ome_zarr_models.v05.image import Image
from pydantic_zarr.v3 import ArraySpec

from .reader import OmeZarrReader


class WriterError(RuntimeError):
    """Raised when the output cannot be written."""


def _nn_resize(mask: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Nearest-neighbour resize a label/binary array to ``target_shape`` (label-preserving)."""
    if mask.shape == tuple(target_shape):
        return mask
    idx = np.ix_(
        *[
            np.clip((np.arange(t) * s // max(t, 1)), 0, s - 1)
            for s, t in zip(mask.shape, target_shape, strict=True)
        ]
    )
    return mask[idx]


def _level_transforms(reader: OmeZarrReader) -> tuple[list[list[float]], list[list[float] | None]]:
    ms = reader.multiscale_model().attributes.ome.multiscales[0]
    scales: list[list[float]] = []
    translations: list[list[float] | None] = []
    for ds in ms.datasets:
        scale: list[float] | None = None
        translation: list[float] | None = None
        for t in ds.coordinateTransformations:
            if getattr(t, "scale", None) is not None:
                scale = list(t.scale)
            if getattr(t, "translation", None) is not None:
                translation = list(t.translation)
        scales.append(scale if scale is not None else [1.0] * len(reader.axes))
        translations.append(translation)
    return scales, translations


def write_mask(
    output_path: str | Path,
    mask: np.ndarray,
    run_level: int,
    reader: OmeZarrReader,
    run_record: dict,
    overwrite: bool = False,
) -> Path:
    """Write ``mask`` (binary, at ``run_level`` shape) as OME-Zarr v0.5 (FR-002, FR-013, FR-014).

    The output has the same number of levels as the input, with shapes/axes/transforms copied
    from the input; the mask is nearest-neighbour resampled to every level's grid (research R5).
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise WriterError(f"output already exists: {output_path} (use overwrite to replace it)")

    mask = (np.asarray(mask) != 0).astype(np.uint8)
    axes = [Axis(name=name, type="space") for name in reader.axes]
    scales, translations = _level_transforms(reader)

    level_arrays = [_nn_resize(mask, reader.level_shape(i)) for i in range(reader.n_levels)]
    specs = [ArraySpec.from_array(a, dimension_names=list(reader.axes)) for a in level_arrays]

    tmp_path = output_path.with_name(output_path.name + ".tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    try:
        image = Image.new(
            array_specs=specs,
            paths=[str(i) for i in range(reader.n_levels)],
            axes=axes,
            scales=scales,
            translations=translations,
        )
        image.to_zarr(zarr.storage.LocalStore(str(tmp_path)), path="")
        for i, arr in enumerate(level_arrays):
            zarr.open_array(store=str(tmp_path), path=str(i), mode="r+")[:] = arr
        (tmp_path / "run_record.json").write_text(json.dumps(run_record, indent=2))

        if output_path.exists():
            shutil.rmtree(output_path)
        tmp_path.rename(output_path)
    except Exception:
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
        raise
    return output_path
