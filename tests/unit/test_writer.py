"""Unit tests for the OME-Zarr writer (T010)."""

from __future__ import annotations

import numpy as np
import pytest

from organ_masker_lite.io.reader import OmeZarrReader
from organ_masker_lite.io.validate import validate_ome_zarr
from organ_masker_lite.io.writer import WriterError, _nn_resize, write_mask

from ..conftest import write_ome_zarr


def _reader(tmp_path, n_levels=3):
    vol = np.zeros((8, 16, 16), np.uint8)
    vol[2:6, 4:12, 4:12] = 200
    return OmeZarrReader(write_ome_zarr(tmp_path / "in.ome.zarr", vol, n_levels=n_levels))


def test_nn_resize_preserves_labels_and_shape():
    mask = np.zeros((8, 16, 16), bool)
    mask[2:6, 4:12, 4:12] = True
    out = _nn_resize(mask, (4, 8, 8))
    assert out.shape == (4, 8, 8)
    assert out.dtype == bool
    assert out.any()


def test_write_matches_input_level_count_and_validates(tmp_path):
    reader = _reader(tmp_path, n_levels=3)
    mask = np.zeros((8, 16, 16), bool)
    mask[2:6, 4:12, 4:12] = True
    out = write_mask(tmp_path / "out.ome.zarr", mask, 0, reader, {"k": "v"})
    out_reader = OmeZarrReader(out)
    assert out_reader.n_levels == reader.n_levels == 3
    assert out_reader.level_shape(0) == (8, 16, 16)
    validate_ome_zarr(out)  # no raise
    assert (out / "run_record.json").exists()
    assert out_reader.read_level(0).max() == 1  # binary 0/1


def test_overwrite_guard(tmp_path):
    reader = _reader(tmp_path)
    mask = np.ones((8, 16, 16), bool)
    write_mask(tmp_path / "out.ome.zarr", mask, 0, reader, {})
    with pytest.raises(WriterError):
        write_mask(tmp_path / "out.ome.zarr", mask, 0, reader, {})
    # with overwrite it succeeds
    write_mask(tmp_path / "out.ome.zarr", mask, 0, reader, {}, overwrite=True)
