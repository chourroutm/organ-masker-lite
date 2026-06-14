"""CLI contract tests for `organ-masker-lite mask` (T029, contracts/cli.md C-CLI-1/4).

Exercised with the deterministic ``stub`` backend so the full arg-schema, exit-code, overwrite-guard
and no-partial-output behaviour is covered without torch/GPU/weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from organ_masker_lite.cli import main
from organ_masker_lite.io.reader import OmeZarrReader

from ..conftest import write_ome_zarr


def _write_prompts(path: Path, *, points=None, box=None) -> Path:
    obj: dict = {"obj_id": 0}
    if points is not None:
        obj["points"] = points
    if box is not None:
        obj["box"] = box
    path.write_text(json.dumps({"objects": [obj]}))
    return path


@pytest.fixture
def valid_input(tmp_path):
    vol = np.zeros((8, 16, 16), np.uint8)
    vol[2:6, 4:12, 4:12] = 200
    return write_ome_zarr(tmp_path / "in.ome.zarr", vol)


def _single_point_prompts(tmp_path) -> Path:
    return _write_prompts(tmp_path / "p.json", points=[{"frame": 4, "xy": [8, 8], "label": 1}])


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["mask", "--help"])
    assert exc.value.code == 0


def test_usage_error_exits_two():
    # missing required positional/option -> argparse usage error
    with pytest.raises(SystemExit) as exc:
        main(["mask"])
    assert exc.value.code == 2


def test_success_writes_valid_output_same_levels(valid_input, tmp_path):
    prompts = _single_point_prompts(tmp_path)
    out = tmp_path / "out.ome.zarr"
    code = main(
        [
            "mask",
            str(valid_input),
            str(out),
            "--prompts",
            str(prompts),
            "--backend",
            "stub",
            "--level",
            "0",
        ]
    )
    assert code == 0
    assert OmeZarrReader(out).n_levels == OmeZarrReader(valid_input).n_levels


def test_invalid_input_store_exits_one_no_output(tmp_path):
    prompts = _single_point_prompts(tmp_path)
    out = tmp_path / "out.ome.zarr"
    code = main(
        [
            "mask",
            str(tmp_path / "missing.ome.zarr"),
            str(out),
            "--prompts",
            str(prompts),
            "--backend",
            "stub",
            "--level",
            "0",
        ]
    )
    assert code == 1
    assert not out.exists()


def test_missing_level_exits_one_no_output(valid_input, tmp_path):
    prompts = _single_point_prompts(tmp_path)
    out = tmp_path / "out.ome.zarr"
    code = main(
        [
            "mask",
            str(valid_input),
            str(out),
            "--prompts",
            str(prompts),
            "--backend",
            "stub",
            "--level",
            "99",
        ]
    )
    assert code == 1
    assert not out.exists()


def test_empty_prompts_exits_one(valid_input, tmp_path):
    prompts = _write_prompts(tmp_path / "empty.json", points=[])
    out = tmp_path / "out.ome.zarr"
    code = main(
        [
            "mask",
            str(valid_input),
            str(out),
            "--prompts",
            str(prompts),
            "--backend",
            "stub",
            "--level",
            "0",
        ]
    )
    assert code == 1
    assert not out.exists()


def test_out_of_bounds_prompt_exits_one(valid_input, tmp_path):
    prompts = _write_prompts(
        tmp_path / "oob.json", points=[{"frame": 4, "xy": [999, 999], "label": 1}]
    )
    out = tmp_path / "out.ome.zarr"
    code = main(
        [
            "mask",
            str(valid_input),
            str(out),
            "--prompts",
            str(prompts),
            "--backend",
            "stub",
            "--level",
            "0",
        ]
    )
    assert code == 1
    assert not out.exists()


def test_existing_output_without_overwrite_exits_one(valid_input, tmp_path):
    prompts = _single_point_prompts(tmp_path)
    out = tmp_path / "out.ome.zarr"
    out.mkdir()
    (out / "sentinel.txt").write_text("keep me")
    code = main(
        [
            "mask",
            str(valid_input),
            str(out),
            "--prompts",
            str(prompts),
            "--backend",
            "stub",
            "--level",
            "0",
        ]
    )
    assert code == 1
    # existing content untouched
    assert (out / "sentinel.txt").read_text() == "keep me"


def test_overwrite_flag_allows_replacement(valid_input, tmp_path):
    prompts = _single_point_prompts(tmp_path)
    out = tmp_path / "out.ome.zarr"
    out.mkdir()
    (out / "stale.txt").write_text("old")
    code = main(
        [
            "mask",
            str(valid_input),
            str(out),
            "--prompts",
            str(prompts),
            "--backend",
            "stub",
            "--level",
            "0",
            "--overwrite",
        ]
    )
    assert code == 0
    assert OmeZarrReader(out).n_levels == OmeZarrReader(valid_input).n_levels
