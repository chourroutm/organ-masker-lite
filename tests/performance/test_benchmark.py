"""Performance benchmark guarding the plan.md / Principle IV targets (T056).

These run with the deterministic stub backend (no GPU), so they assert the targets that hold
without a real SAM backend:

- Peak host RAM stays well under the ceiling and its per-voxel cost is bounded and independent of
  slice count along the sweep axis -- the frame and vote intermediates are memmap-backed
  (out-of-core), so they add no resident RAM that grows with depth.
- A coarser pyramid level strictly reduces end-to-end runtime versus a finer one (SC-006).

The 512x512x512-under-5-minutes-on-GPU runtime budget requires a real backend and is out of scope
for the stub suite.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pytest

from organ_masker_lite.config import RunConfig
from organ_masker_lite.engine.pipeline import compute_mask, run_masking
from organ_masker_lite.prompts.model import Prompt, PromptSet

from ..conftest import StubBackend, write_ome_zarr

pytestmark = pytest.mark.slow

# Generous per-voxel ceiling: the pre-out-of-core implementation (whole-volume float64 copy) used
# ~18 bytes/voxel; the memmap design stays well under this.
MAX_BYTES_PER_VOXEL = 12.0


def _peak_bytes(store, prompts, level=0):
    tracemalloc.start()
    compute_mask(
        store, prompts, RunConfig(backend="stub", axes=["z"], level=level), backend=StubBackend()
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def _blob_store(tmp_path, depth, hw=192, n_levels=1):
    vol = np.zeros((depth, hw, hw), np.uint8)
    vol[depth // 4 : depth * 3 // 4, hw // 4 : hw * 3 // 4, hw // 4 : hw * 3 // 4] = 200
    return write_ome_zarr(tmp_path / f"d{depth}.ome.zarr", vol, n_levels=n_levels)


def _center_prompt(depth, hw=192):
    return PromptSet(
        [Prompt(frame_index=depth // 2, point_coords=[[hw // 2, hw // 2]], point_labels=[1])]
    )


def test_peak_ram_bounded_and_depth_independent(tmp_path):
    shallow, deep = 64, 128  # deep has 2x the slice count, same in-plane size
    peak_shallow = _peak_bytes(_blob_store(tmp_path, shallow), _center_prompt(shallow))
    peak_deep = _peak_bytes(_blob_store(tmp_path, deep), _center_prompt(deep))

    bpv_shallow = peak_shallow / (shallow * 192 * 192)
    bpv_deep = peak_deep / (deep * 192 * 192)

    # well under the 4 GB ceiling for these reference-proportional volumes
    assert peak_deep < 256 * 1024**2
    # per-voxel RAM is bounded (intermediates are out-of-core, not held in RAM)
    assert bpv_deep < MAX_BYTES_PER_VOXEL
    # and does not grow with slice count along the sweep axis
    assert bpv_deep <= bpv_shallow * 1.2


def test_coarser_level_strictly_faster(tmp_path):
    store = _blob_store(tmp_path, 32, hw=320, n_levels=2)  # levels (32,320,320) and (16,160,160)
    fine = _center_prompt(32, hw=320)
    coarse = PromptSet([Prompt(frame_index=8, point_coords=[[80, 80]], point_labels=[1])])

    def best(prompts, level, tag):
        cfg = RunConfig(backend="stub", axes=["z"], level=level)
        run_masking(store, tmp_path / f"{tag}_warm.ome.zarr", prompts, cfg, backend=StubBackend())
        best_t = float("inf")
        for i in range(5):
            start = time.perf_counter()
            run_masking(
                store, tmp_path / f"{tag}_{i}.ome.zarr", prompts, cfg, backend=StubBackend()
            )
            best_t = min(best_t, time.perf_counter() - start)
        return best_t

    fine_t = best(fine, 0, "fine")
    coarse_t = best(coarse, 1, "coarse")
    assert coarse_t < fine_t, f"coarse {coarse_t:.4f}s not faster than fine {fine_t:.4f}s"
