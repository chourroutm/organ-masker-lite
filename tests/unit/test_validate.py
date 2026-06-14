"""Unit tests for OME-Zarr input validation via the ome-zarr-models CLI (T011)."""

from __future__ import annotations

import numpy as np
import pytest

from organ_masker_lite.io.validate import ValidationError, validate_ome_zarr

from ..conftest import write_ome_zarr


def test_valid_store_passes(tmp_path):
    vol = np.zeros((8, 16, 16), np.uint8)
    vol[2:6, 4:12, 4:12] = 200
    store = write_ome_zarr(tmp_path / "in.ome.zarr", vol)
    validate_ome_zarr(store)  # no raise


def test_missing_path_raises(tmp_path):
    with pytest.raises(ValidationError):
        validate_ome_zarr(tmp_path / "nope.ome.zarr")


def test_non_ome_store_raises(tmp_path):
    junk = tmp_path / "junk"
    junk.mkdir()
    (junk / "hello.txt").write_text("not a zarr")
    with pytest.raises(ValidationError):
        validate_ome_zarr(junk)
