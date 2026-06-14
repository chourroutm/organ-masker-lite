"""Real-SAM2 smoke test (T034). Skipped unless torch + sam2 + weights are available.

Marked ``real_backend`` so it is excluded from the default deterministic suite; run explicitly with
``pytest -m real_backend``. It only checks that the SAM2 adapter runs end-to-end and writes a valid
OME-Zarr mask with the same level count as the input -- not segmentation quality.
"""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.real_backend

_HAVE_SAM2 = (
    importlib.util.find_spec("torch") is not None and importlib.util.find_spec("sam2") is not None
)


@pytest.mark.skipif(not _HAVE_SAM2, reason="torch/sam2 not installed")
def test_sam2_backend_end_to_end(single_blob_zarr, tmp_path):
    from organ_masker_lite.backends.registry import get_backend
    from organ_masker_lite.config import RunConfig
    from organ_masker_lite.engine.pipeline import run_masking
    from organ_masker_lite.io.reader import OmeZarrReader
    from organ_masker_lite.prompts.model import Prompt, PromptSet

    store, (cx, cy), frame = single_blob_zarr
    prompts = PromptSet([Prompt(frame_index=frame, point_coords=[[cx, cy]], point_labels=[1])])
    cfg = RunConfig(backend="sam2", axes=["z"], level=0)

    try:
        backend = get_backend("sam2")
    except ImportError as exc:  # weights/runtime missing despite importable packages
        pytest.skip(f"sam2 backend unavailable: {exc}")

    out = run_masking(store, tmp_path / "out.ome.zarr", prompts, cfg, backend=backend)

    in_reader = OmeZarrReader(store)
    out_reader = OmeZarrReader(out)
    assert out_reader.n_levels == in_reader.n_levels
    assert out_reader.read_level(0).any()
