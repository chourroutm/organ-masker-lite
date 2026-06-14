"""US1 second feature: a bounding-box prompt yields foreground within the box (T027, C-CLI-2)."""

from __future__ import annotations

import numpy as np

from organ_masker_lite.config import RunConfig
from organ_masker_lite.engine.pipeline import run_masking
from organ_masker_lite.io.reader import OmeZarrReader
from organ_masker_lite.prompts.model import Prompt, PromptSet


def test_box_mask_foreground_within_box(single_blob_zarr, stub_backend, tmp_path):
    store, _center, frame = single_blob_zarr
    # blob occupies x in [4, 12), y in [4, 12); draw a box that encloses it
    box = [4.0, 4.0, 12.0, 12.0]
    prompts = PromptSet(
        [Prompt(frame_index=frame, point_coords=np.empty((0, 2)), point_labels=[], box=box)]
    )
    cfg = RunConfig(backend="stub", axes=["z"], level=0)

    out = run_masking(store, tmp_path / "out.ome.zarr", prompts, cfg, backend=stub_backend)

    mask = OmeZarrReader(out).read_level(0)
    assert mask.any()
    # every foreground voxel lies within the box's in-plane (x, y) extent
    _z, ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = box
    assert (xs >= x0).all() and (xs < x1).all()
    assert (ys >= y0).all() and (ys < y1).all()
