"""Backend comparison harness (T035, SC-010, C-CLI-5).

Verifies that the same input and prompts can be run through two different backends by changing
*only* the backend selection, and that both produce valid OME-Zarr v0.5 masks with the input's
level count -- enabling a like-for-like comparison (IoU).

The deterministic test uses two registered stub backends (``stub`` and ``stub_dilate``) so the
harness runs in CI. The real SAM2-vs-SAM3 comparison is the skip-guarded counterpart below.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from organ_masker_lite.cli import main
from organ_masker_lite.io.reader import OmeZarrReader

from ..conftest import write_ome_zarr


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def _run(input_path: Path, out: Path, prompts_file: Path, backend: str) -> int:
    """Run the CLI changing only ``--backend``; returns the process exit code."""
    return main(
        [
            "mask",
            str(input_path),
            str(out),
            "--prompts",
            str(prompts_file),
            "--backend",
            backend,
            "--level",
            "0",
        ]
    )


@pytest.fixture
def input_and_prompts(tmp_path):
    vol = np.zeros((8, 16, 16), np.uint8)
    vol[2:6, 4:12, 4:12] = 200
    store = write_ome_zarr(tmp_path / "in.ome.zarr", vol)
    prompts = tmp_path / "p.json"
    prompts.write_text(
        json.dumps({"objects": [{"obj_id": 0, "points": [{"frame": 4, "xy": [8, 8], "label": 1}]}]})
    )
    return store, prompts


def test_two_backends_selectable_produce_valid_comparable_masks(input_and_prompts, tmp_path):
    store, prompts = input_and_prompts
    out_a = tmp_path / "a.ome.zarr"
    out_b = tmp_path / "b.ome.zarr"

    assert _run(store, out_a, prompts, "stub") == 0
    assert _run(store, out_b, prompts, "stub_dilate") == 0

    rin = OmeZarrReader(store)
    ra = OmeZarrReader(out_a)
    rb = OmeZarrReader(out_b)
    # both outputs are valid OME-Zarr with the same level count as the input
    assert ra.n_levels == rin.n_levels == rb.n_levels

    mask_a = ra.read_level(0)
    mask_b = rb.read_level(0)
    assert mask_a.shape == mask_b.shape
    assert mask_a.any() and mask_b.any()

    # like-for-like comparison: overlapping but not necessarily identical
    iou = _mask_iou(mask_a, mask_b)
    assert 0.0 < iou <= 1.0


_HAVE_BOTH = all(importlib.util.find_spec(m) is not None for m in ("torch", "sam2", "sam3"))


@pytest.mark.real_backend
@pytest.mark.skipif(not _HAVE_BOTH, reason="torch/sam2/sam3 not installed")
def test_sam2_vs_sam3_like_for_like(input_and_prompts, tmp_path):
    store, prompts = input_and_prompts
    out2 = tmp_path / "sam2.ome.zarr"
    out3 = tmp_path / "sam3.ome.zarr"

    assert _run(store, out2, prompts, "sam2") == 0
    assert _run(store, out3, prompts, "sam3") == 0

    rin = OmeZarrReader(store)
    r2 = OmeZarrReader(out2)
    r3 = OmeZarrReader(out3)
    assert r2.n_levels == rin.n_levels == r3.n_levels

    mask2 = r2.read_level(0)
    mask3 = r3.read_level(0)
    assert mask2.shape == mask3.shape
    # the comparison metric is available for the user to inspect
    assert 0.0 <= _mask_iou(mask2, mask3) <= 1.0
