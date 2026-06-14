"""Shared test fixtures: a synthetic OME-Zarr v0.5 volume and a deterministic stub backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr
from ome_zarr_models.v05.axes import Axis
from ome_zarr_models.v05.image import Image
from pydantic_zarr.v3 import ArraySpec
from scipy.ndimage import binary_dilation, label

from organ_masker_lite.backends.registry import register_backend
from organ_masker_lite.prompts.model import PromptSet


def write_ome_zarr(path: Path, level0: np.ndarray, n_levels: int = 2) -> Path:
    """Write a minimal valid OME-Zarr v0.5 multiscale image (axes z, y, x)."""
    arrays = [level0]
    for _ in range(1, n_levels):
        arrays.append(arrays[-1][::2, ::2, ::2].copy())
    axes = [Axis(name=n, type="space") for n in ("z", "y", "x")]
    specs = [ArraySpec.from_array(a, dimension_names=["z", "y", "x"]) for a in arrays]
    image = Image.new(
        array_specs=specs,
        paths=[str(i) for i in range(n_levels)],
        axes=axes,
        scales=[[2.0**i] * 3 for i in range(n_levels)],
        translations=[None] * n_levels,
    )
    image.to_zarr(zarr.storage.LocalStore(str(path)), path="")
    for i, arr in enumerate(arrays):
        zarr.open_array(store=str(path), path=str(i), mode="r+")[:] = arr
    return path


@pytest.fixture
def single_blob_zarr(tmp_path):
    """A volume with one bright cuboid blob; returns (path, blob_center_xy, frame_index)."""
    vol = np.zeros((8, 16, 16), dtype=np.uint8)
    vol[2:6, 4:12, 4:12] = 200
    path = write_ome_zarr(tmp_path / "input.ome.zarr", vol)
    # center of the blob on frame z=4: (x, y) = (8, 8)
    return path, (8.0, 8.0), 4


@pytest.fixture
def two_blob_zarr(tmp_path):
    """A volume with two separate blobs; returns (path, blobA_xy, blobB_xy, frame_index)."""
    vol = np.zeros((8, 24, 24), dtype=np.uint8)
    vol[2:6, 3:9, 3:9] = 200  # blob A
    vol[2:6, 15:21, 15:21] = 200  # blob B (separate)
    path = write_ome_zarr(tmp_path / "two.ome.zarr", vol)
    return path, (6.0, 6.0), (18.0, 18.0), 4


class StubBackend:
    """Deterministic, dependency-light backend: threshold + connected-component selection.

    Foreground = connected components (3D) of the thresholded frame stack that contain a positive
    point or overlap a box, minus components that contain a negative point.
    """

    name = "stub"

    def segment_video(self, frames: np.ndarray, prompts: PromptSet) -> np.ndarray:
        gray = np.asarray(frames)[..., 0]
        threshold = max(1, int(gray.max()) // 2)
        candidate = gray >= threshold
        labels, _n = label(candidate)
        selected: set[int] = set()
        deselected: set[int] = set()
        for p in prompts.prompts:
            f = p.frame_index
            for (x, y), lbl in zip(p.point_coords, p.point_labels, strict=True):
                comp = int(labels[f, int(round(y)), int(round(x))])
                if comp == 0:
                    continue
                (selected if lbl == 1 else deselected).add(comp)
            if p.box is not None:
                x0, y0, x1, y1 = p.box.astype(int)
                for comp in np.unique(labels[f, y0:y1, x0:x1]):
                    if comp != 0:
                        selected.add(int(comp))
        mask = np.isin(labels, list(selected)) if selected else np.zeros_like(candidate, bool)
        if deselected:
            mask &= ~np.isin(labels, list(deselected))
        return mask


class StubBackendDilate(StubBackend):
    """A second deterministic backend: the stub result dilated by one voxel.

    Used to exercise the SAM2-vs-SAM3 comparison harness with two distinct, selectable backends
    that produce overlapping-but-different masks, without torch/GPU/weights.
    """

    name = "stub_dilate"

    def segment_video(self, frames: np.ndarray, prompts: PromptSet) -> np.ndarray:
        base = super().segment_video(frames, prompts)
        return binary_dilation(base) if base.any() else base


@pytest.fixture
def stub_backend():
    return StubBackend()


# Register the stubs so CLI/integration tests can select them via --backend.
register_backend("stub", lambda options: StubBackend())
register_backend("stub_dilate", lambda options: StubBackendDilate())
