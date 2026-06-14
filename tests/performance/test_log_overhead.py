"""T024: logging adds < 50 ms per invocation and does not scale with volume size (feature 002)."""

from __future__ import annotations

import time

from organ_masker_lite.config import LogConfig
from organ_masker_lite.input_log import log_invocation
from organ_masker_lite.prompts.model import Prompt, PromptSet

BUDGET_S = 0.050


def _time_one(tmp_path, level_shape):
    """Time a full log lifecycle (open -> config -> prompts -> finish) for a given volume shape."""
    prompts = PromptSet([Prompt(frame_index=0, point_coords=[[1.0, 1.0]], point_labels=[1])])
    config = {"backend": "stub", "level": 0, "level_shape": list(level_shape)}
    cfg = LogConfig(log_dir=tmp_path)
    start = time.perf_counter()
    with log_invocation("organ-masker-lite mask in out", cfg, arguments={"backend": "stub"}) as log:
        log.record_config(config)
        log.record_prompts(prompts, source="file")
    return time.perf_counter() - start


def test_logging_overhead_under_budget(tmp_path):
    best = min(_time_one(tmp_path, (32, 256, 256)) for _ in range(5))
    assert best < BUDGET_S, f"logging overhead {best * 1000:.1f} ms over {BUDGET_S * 1000:.0f} ms"


def test_logging_overhead_independent_of_volume_size(tmp_path):
    # Logging never reads volume data, so a 1000x larger volume must not cost more.
    small = min(_time_one(tmp_path, (16, 64, 64)) for _ in range(5))
    large = min(_time_one(tmp_path, (256, 1024, 1024)) for _ in range(5))
    assert small < BUDGET_S
    assert large < BUDGET_S
