"""Unit tests for the SAM2-compatible prompt model (T006)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from organ_masker_lite.prompts.model import (
    Prompt,
    PromptError,
    PromptSet,
    load_prompt_file,
)


def test_point_coords_and_labels_must_match():
    with pytest.raises(PromptError):
        Prompt(frame_index=0, point_coords=[[1, 2], [3, 4]], point_labels=[1])


def test_labels_must_be_zero_or_one():
    with pytest.raises(PromptError):
        Prompt(frame_index=0, point_coords=[[1, 2]], point_labels=[2])


def test_positive_and_negative_coords_split():
    p = Prompt(frame_index=0, point_coords=[[1, 1], [2, 2]], point_labels=[1, 0])
    assert p.positive_coords.tolist() == [[1, 1]]
    assert p.negative_coords.tolist() == [[2, 2]]


def test_validate_requires_a_positive_prompt():
    ps = PromptSet([Prompt(frame_index=0, point_coords=[[1, 1]], point_labels=[0])])
    with pytest.raises(PromptError):
        ps.validate(level_shape=(4, 8, 8), axis_index=0)


def test_validate_rejects_out_of_bounds_point():
    ps = PromptSet([Prompt(frame_index=0, point_coords=[[99, 1]], point_labels=[1])])
    with pytest.raises(PromptError):
        ps.validate(level_shape=(4, 8, 8), axis_index=0)


def test_validate_rejects_out_of_range_frame():
    ps = PromptSet([Prompt(frame_index=99, point_coords=[[1, 1]], point_labels=[1])])
    with pytest.raises(PromptError):
        ps.validate(level_shape=(4, 8, 8), axis_index=0)


def test_validate_accepts_in_bounds_box():
    ps = PromptSet(
        [
            Prompt(
                frame_index=0,
                point_coords=np.empty((0, 2)),
                point_labels=np.empty((0,), int),
                box=[1, 1, 5, 5],
            )
        ]
    )
    ps.validate(level_shape=(4, 8, 8), axis_index=0)  # no raise


def test_load_prompt_file_single_point(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(
        json.dumps({"objects": [{"obj_id": 0, "points": [{"frame": 4, "xy": [8, 8], "label": 1}]}]})
    )
    ps = load_prompt_file(f)
    assert len(ps.prompts) == 1
    assert ps.prompts[0].frame_index == 4
    assert ps.has_positive()


def test_load_prompt_file_box(tmp_path):
    f = tmp_path / "b.json"
    f.write_text(json.dumps({"objects": [{"box": {"frame": 4, "xyxy": [3, 3, 13, 13]}}]}))
    ps = load_prompt_file(f)
    assert ps.prompts[0].box.tolist() == [3, 3, 13, 13]
    assert ps.has_positive()
