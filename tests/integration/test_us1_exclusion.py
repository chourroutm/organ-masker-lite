"""US1 third feature: an exclusion point (label 0) removes that region (T028, C-CLI-3)."""

from __future__ import annotations

from organ_masker_lite.config import RunConfig
from organ_masker_lite.engine.pipeline import run_masking
from organ_masker_lite.io.reader import OmeZarrReader
from organ_masker_lite.prompts.model import Prompt, PromptSet


def test_exclusion_point_removes_region(two_blob_zarr, stub_backend, tmp_path):
    store, (ax, ay), (bx, by), frame = two_blob_zarr
    # select both blobs, then exclude blob B with a negative point
    prompts = PromptSet(
        [
            Prompt(
                frame_index=frame,
                point_coords=[[ax, ay], [bx, by], [bx, by]],
                point_labels=[1, 1, 0],
            )
        ]
    )
    cfg = RunConfig(backend="stub", axes=["z"], level=0)

    out = run_masking(store, tmp_path / "out.ome.zarr", prompts, cfg, backend=stub_backend)

    mask = OmeZarrReader(out).read_level(0)
    assert mask.any()
    # blob A retained, blob B removed by the exclusion point
    assert mask[frame, int(ay), int(ax)] == 1
    assert mask[frame, int(by), int(bx)] == 0
