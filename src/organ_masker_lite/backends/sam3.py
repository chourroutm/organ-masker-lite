"""SAM3 video-predictor backend adapter.

A first-class backend alongside SAM2 (FR-018), implementing the same
``VideoSegmenterBackend`` shape so the two are interchangeable via ``--backend`` (SC-010).
Imports of ``torch``/``sam3`` are deferred to construction so the package (and its tests with
the stub backend) work without those heavy, optional dependencies installed. Install with the
``[sam3]`` extra to use this backend.

Like the SAM2 adapter, the predictor is built from a Hydra config name and a checkpoint file
(FR-019/FR-020): the checkpoint is resolved against the run's model directory and auto-downloaded
on first use unless downloads are disabled. Override the config name / checkpoint via
``ORGAN_MASKER_SAM3_CONFIG`` / ``ORGAN_MASKER_SAM3_CHECKPOINT`` if your installed ``sam3`` build
ships different config paths or you want a different variant.
"""

from __future__ import annotations

import os
import tempfile
import urllib.request
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from ..config import RunConfig
from ..prompts.model import PromptSet

#: Default SAM3 Hiera-Large variant. The config name is resolved by sam3's bundled Hydra search
#: path; the checkpoint filename lives in the run's model directory. Both are overridable via the
#: environment because the released sam3 config paths/URLs may differ from these defaults.
_DEFAULT_CONFIG = "configs/sam3/sam3_hiera_l.yaml"
_DEFAULT_CHECKPOINT = "sam3_hiera_large.pt"
_CHECKPOINT_URL = os.environ.get("ORGAN_MASKER_SAM3_CHECKPOINT_URL", "")

_CONFIG_ENV = "ORGAN_MASKER_SAM3_CONFIG"
_CHECKPOINT_ENV = "ORGAN_MASKER_SAM3_CHECKPOINT"


class Sam3Backend:
    """Adapter over SAM3's video predictor (``build_sam3_video_predictor``)."""

    name = "sam3"

    def __init__(self, config: RunConfig | None = None, **_: object):
        self._config = config if config is not None else RunConfig(backend="sam3")
        try:
            import torch
            from sam3.build_sam import build_sam3_video_predictor
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "the SAM3 backend requires the '[sam3]' extra (torch + sam3). "
                "Install it, or use a different backend."
            ) from exc
        self._torch = torch
        self._build = build_sam3_video_predictor
        self._config_name = os.environ.get(_CONFIG_ENV, _DEFAULT_CONFIG)
        self._predictor = None  # built lazily on first segmentation, then reused

    def _resolve_checkpoint(self) -> Path:
        """Resolve the checkpoint path, downloading it on first use unless disabled (FR-019/020)."""
        model_dir = self._config.resolved_model_dir()
        filename = os.environ.get(_CHECKPOINT_ENV, _DEFAULT_CHECKPOINT)
        ckpt = model_dir / filename
        if ckpt.exists():
            return ckpt
        if not self._config.allow_download:
            raise FileNotFoundError(
                f"SAM3 checkpoint not found at {ckpt} and downloads are disabled "
                f"(--no-download); place the weights there or allow downloads."
            )
        if not _CHECKPOINT_URL:
            raise FileNotFoundError(
                f"SAM3 checkpoint not found at {ckpt} and no download URL is configured; "
                f"set ORGAN_MASKER_SAM3_CHECKPOINT_URL or place the weights there."
            )
        model_dir.mkdir(parents=True, exist_ok=True)
        tmp = ckpt.with_suffix(ckpt.suffix + ".part")
        urllib.request.urlretrieve(_CHECKPOINT_URL, tmp)  # noqa: S310 (operator-configured URL)
        tmp.replace(ckpt)
        return ckpt

    def _get_predictor(self):
        """Build the video predictor once (config + checkpoint + device) and cache it."""
        if self._predictor is None:
            device = "cuda" if self._torch.cuda.is_available() else "cpu"
            ckpt = self._resolve_checkpoint()
            self._predictor = self._build(self._config_name, str(ckpt), device=device)
        return self._predictor

    def segment_video(
        self, frames: np.ndarray, prompts: PromptSet
    ) -> np.ndarray:  # pragma: no cover
        """Propagate a mask across ``frames`` using SAM3's video predictor.

        Mirrors the SAM2 adapter: frames are materialised to a temporary JPEG directory, prompts
        are added on their frames via ``add_new_points_or_box`` using the encoder's
        ``(coords, labels, box)`` convention, then ``propagate_in_video`` fills the stack.
        """
        from PIL import Image as PILImage

        torch = self._torch
        n_frames, height, width = frames.shape[:3]
        predictor = self._get_predictor()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with tempfile.TemporaryDirectory() as frame_dir:
            for i in range(n_frames):
                PILImage.fromarray(frames[i]).save(Path(frame_dir) / f"{i:05d}.jpg")
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
            )
            out = np.zeros((n_frames, height, width), dtype=bool)
            with torch.inference_mode(), autocast:
                state = predictor.init_state(video_path=frame_dir)
                obj_id = 0
                for p in prompts.prompts:
                    kwargs: dict = {
                        "inference_state": state,
                        "frame_idx": p.frame_index,
                        "obj_id": obj_id,
                    }
                    if p.point_coords.size:
                        kwargs["points"] = p.point_coords.astype("float32")
                        kwargs["labels"] = p.point_labels.astype("int32")
                    if p.box is not None:
                        kwargs["box"] = p.box.astype("float32")
                    predictor.add_new_points_or_box(**kwargs)
                for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
                    out[frame_idx] = (mask_logits[0] > 0).cpu().numpy().reshape(height, width)
        return out
