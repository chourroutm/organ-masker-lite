"""SAM3 video-predictor backend adapter.

A first-class backend alongside SAM2 (FR-018), implementing the same
``VideoSegmenterBackend`` shape so the two are interchangeable via ``--backend`` (SC-010).
Imports of ``torch``/``sam3`` are deferred to construction so the package (and its tests with
the stub backend) work without those heavy, optional dependencies installed. Install with the
``[sam3]`` extra to use this backend.

Unlike the SAM2 adapter, SAM3's video predictor is not built from a Hydra config name: it is
constructed directly via ``sam3.model_builder.build_sam3_video_predictor(gpus_to_use=...)`` and
resolves its own weights from the Hugging Face Hub (``facebook/sam3``) on first use (FR-019/020).
Set ``ORGAN_MASKER_SAM3_CHECKPOINT`` to a local ``.pt`` path (absolute, or relative to the run's
model directory) to use a specific checkpoint file instead of the Hub default. Inference uses
SAM3's request-dispatch session API (``start_session`` / ``add_prompt`` / ``propagate_in_video``).
"""

from __future__ import annotations

import os
import tempfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from ..config import RunConfig
from ..prompts.model import PromptSet

#: Optional explicit checkpoint. By default SAM3 resolves/downloads its weights from the Hugging
#: Face Hub; set this to a local ``.pt`` path (absolute or relative to the run's model directory)
#: to pin a specific checkpoint file.
_CHECKPOINT_ENV = "ORGAN_MASKER_SAM3_CHECKPOINT"


class Sam3Backend:
    """Adapter over SAM3's video predictor (``build_sam3_video_predictor``)."""

    name = "sam3"

    def __init__(self, config: RunConfig | None = None, **_: object):
        self._config = config if config is not None else RunConfig(backend="sam3")
        try:
            import torch
            from sam3.model_builder import build_sam3_video_predictor
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "the SAM3 backend requires the '[sam3]' extra (torch + sam3). "
                "Install it, or use a different backend."
            ) from exc
        self._torch = torch
        self._build = build_sam3_video_predictor
        self._predictor = None  # built lazily on first segmentation, then reused

    def _resolve_checkpoint(self) -> str | None:
        """Resolve an optional local checkpoint path (FR-019/020).

        Returns ``None`` to let SAM3 download its default weights from the Hugging Face Hub.
        If ``ORGAN_MASKER_SAM3_CHECKPOINT`` names a file (absolute, or relative to the run's model
        directory) that exists, that path is used. When downloads are disabled (``--no-download``)
        and no usable local checkpoint is available, the Hub is forced offline so cached weights are
        used (or a clear Hub error is raised) rather than silently fetching over the network.
        """
        filename = os.environ.get(_CHECKPOINT_ENV)
        if filename:
            ckpt = Path(filename)
            if not ckpt.is_absolute():
                ckpt = self._config.resolved_model_dir() / filename
            if ckpt.exists():
                return str(ckpt)
            if not self._config.allow_download:
                raise FileNotFoundError(
                    f"SAM3 checkpoint not found at {ckpt} and downloads are disabled "
                    f"(--no-download); place the weights there or allow downloads."
                )
        if not self._config.allow_download:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        return None

    def _get_predictor(self):
        """Build the video predictor once (weights + device) and cache it."""
        if self._predictor is None:
            torch = self._torch
            ckpt = self._resolve_checkpoint()
            gpus = [torch.cuda.current_device()] if torch.cuda.is_available() else None
            kwargs: dict = {"gpus_to_use": gpus}
            if ckpt is not None:
                kwargs["checkpoint_path"] = ckpt
            self._predictor = self._build(**kwargs)
        return self._predictor

    @staticmethod
    def _frame_mask(outputs: dict, height: int, width: int) -> np.ndarray:
        """Combine SAM3's per-object binary masks for one frame into a single foreground mask."""
        combined = np.zeros((height, width), dtype=bool)
        binary = outputs.get("out_binary_masks")
        if binary is None:
            return combined
        for m in binary:
            arr = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
            arr = np.squeeze(arr)
            if arr.shape != (height, width):
                from PIL import Image as PILImage

                arr = np.asarray(
                    PILImage.fromarray(arr.astype(np.uint8)).resize(
                        (width, height), PILImage.NEAREST
                    )
                )
            combined |= arr > 0.5
        return combined

    def segment_video(
        self, frames: np.ndarray, prompts: PromptSet
    ) -> np.ndarray:  # pragma: no cover
        """Propagate a mask across ``frames`` using SAM3's video predictor.

        Frames are materialised to a temporary JPEG directory, a session is opened on that
        directory, prompts are added on their frames via the ``add_prompt`` request, then
        ``propagate_in_video`` streams a per-frame result whose ``out_binary_masks`` are reduced to
        a single boolean foreground mask per frame.
        """
        from PIL import Image as PILImage

        torch = self._torch
        n_frames, height, width = frames.shape[:3]
        predictor = self._get_predictor()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        out = np.zeros((n_frames, height, width), dtype=bool)
        with tempfile.TemporaryDirectory() as frame_dir:
            for i in range(n_frames):
                PILImage.fromarray(frames[i]).save(Path(frame_dir) / f"{i:05d}.jpg")
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
            )
            with torch.inference_mode(), autocast:
                session_id = predictor.handle_request(
                    {"type": "start_session", "resource_path": frame_dir}
                )["session_id"]
                try:
                    for p in prompts.prompts:
                        request: dict = {
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": p.frame_index,
                            "obj_id": 0,
                        }
                        if p.point_coords.size:
                            request["points"] = p.point_coords.astype("float32").tolist()
                            request["point_labels"] = p.point_labels.astype("int32").tolist()
                        if p.box is not None:
                            request["bounding_boxes"] = [p.box.astype("float32").tolist()]
                            request["bounding_box_labels"] = [1]
                        predictor.handle_request(request)
                    for response in predictor.handle_stream_request(
                        {"type": "propagate_in_video", "session_id": session_id}
                    ):
                        out[response["frame_index"]] = self._frame_mask(
                            response["outputs"], height, width
                        )
                finally:
                    predictor.handle_request({"type": "close_session", "session_id": session_id})
        return out
