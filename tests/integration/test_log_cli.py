"""T008: a successful CLI ``mask`` run writes a log with command + config (C-LOG-CLI-1)."""

from __future__ import annotations

import json

from organ_masker_lite.cli import main


def _write_prompts(path, frame, cx, cy):
    path.write_text(
        json.dumps(
            {"objects": [{"obj_id": 0, "points": [{"frame": frame, "xy": [cx, cy], "label": 1}]}]}
        )
    )


def test_successful_cli_run_writes_log(single_blob_zarr, tmp_path):
    store, (cx, cy), frame = single_blob_zarr
    prompts = tmp_path / "p.json"
    _write_prompts(prompts, frame, cx, cy)
    logdir = tmp_path / "logs"
    code = main(
        [
            "mask",
            str(store),
            str(tmp_path / "out.ome.zarr"),
            "--prompts",
            str(prompts),
            "--backend",
            "stub",
            "--level",
            "0",
            "--log-dir",
            str(logdir),
        ]
    )
    assert code == 0
    logs = list(logdir.glob("*.log"))
    assert len(logs) == 1
    text = logs[0].read_text()
    assert "command: organ-masker-lite mask" in text
    assert "effective_config:" in text
    assert "backend: stub" in text
    assert "outcome: succeeded" in text
