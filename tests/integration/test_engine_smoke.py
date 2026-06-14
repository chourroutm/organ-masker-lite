"""Engine smoke test: stub-backend single-axis pipeline end-to-end (T013)."""

from __future__ import annotations

from organ_masker_lite.config import RunConfig
from organ_masker_lite.engine.pipeline import run_masking
from organ_masker_lite.io.reader import OmeZarrReader
from organ_masker_lite.prompts.model import Prompt, PromptSet


def test_pipeline_produces_mask_with_matching_levels(single_blob_zarr, stub_backend, tmp_path):
    store, (cx, cy), frame = single_blob_zarr
    prompts = PromptSet([Prompt(frame_index=frame, point_coords=[[cx, cy]], point_labels=[1])])
    cfg = RunConfig(backend="stub", axes=["z"], level=0)
    out = run_masking(store, tmp_path / "out.ome.zarr", prompts, cfg, backend=stub_backend)

    in_reader = OmeZarrReader(store)
    out_reader = OmeZarrReader(out)
    assert out_reader.n_levels == in_reader.n_levels
    mask = out_reader.read_level(0)
    assert mask.any()
    assert mask.shape == in_reader.level_shape(0)
