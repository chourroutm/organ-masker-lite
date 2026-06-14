"""SAM2 video-predictor backend adapter.

Imports of ``torch``/``sam2`` are deferred to construction so the package (and its tests with
the stub backend) work without those heavy, optional dependencies installed. Install with the
``[sam2]`` extra to use this backend.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ..config import RunConfig
from ..prompts.model import PromptSet


class Sam2Backend:
    """Adapter over SAM2's video predictor (``build_sam2_video_predictor``)."""

    name = "sam2"

    def __init__(self, config: RunConfig | None = None, **_: object):
        self._config = config
        try:
            import torch  # noqa: F401
            from sam2.build_sam import build_sam2_video_predictor  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "the SAM2 backend requires the '[sam2]' extra (torch + sam2). "
                "Install it, or use a different backend."
            ) from exc
        self._build = build_sam2_video_predictor

    def segment_video(
        self, frames: np.ndarray, prompts: PromptSet
    ) -> np.ndarray:  # pragma: no cover
        """Propagate a mask across ``frames`` using SAM2's video predictor.

        Frames are materialised to a temporary JPEG directory (the SAM2-native input form),
        prompts are added on their frames via ``add_new_points_or_box`` using the encoder's
        ``(coords, labels, box)`` convention, then ``propagate_in_video`` fills the stack.
        """
        from PIL import Image as PILImage

        n_frames, height, width = frames.shape[:3]
        predictor = self._build()
        with tempfile.TemporaryDirectory() as frame_dir:
            for i in range(n_frames):
                PILImage.fromarray(frames[i]).save(Path(frame_dir) / f"{i:05d}.jpg")
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
            out = np.zeros((n_frames, height, width), dtype=bool)
            for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
                out[frame_idx] = (mask_logits[0] > 0).cpu().numpy().reshape(height, width)
        return out
