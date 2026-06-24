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

SAM3's text encoder also needs a BPE tokenizer vocab (``bpe_simple_vocab_16e6.txt.gz``). Plain
``pip install`` builds of ``sam3`` may omit this bundled asset, and the path sam3 derives for it
varies by version, so this adapter resolves the vocab itself into the run's model directory
(downloading the standard CLIP vocab on first use unless downloads are disabled) and passes it as
an explicit ``bpe_path``. Override the location with ``ORGAN_MASKER_SAM3_BPE`` (a local path) or the
source with ``ORGAN_MASKER_SAM3_BPE_URL``.
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

#: Optional explicit checkpoint. By default SAM3 resolves/downloads its weights from the Hugging
#: Face Hub; set this to a local ``.pt`` path (absolute or relative to the run's model directory)
#: to pin a specific checkpoint file.
_CHECKPOINT_ENV = "ORGAN_MASKER_SAM3_CHECKPOINT"

#: BPE tokenizer vocab sam3's text encoder requires. The standard CLIP vocab (sam3's tokenizer is
#: CLIP-derived); resolved into the run's model directory unless overridden. ``_BPE_ENV`` pins a
#: local path; ``_BPE_URL`` overrides the download source.
_BPE_FILENAME = "bpe_simple_vocab_16e6.txt.gz"
_BPE_ENV = "ORGAN_MASKER_SAM3_BPE"
_BPE_URL = os.environ.get(
    "ORGAN_MASKER_SAM3_BPE_URL",
    "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz",
)

#: Guidance shown when SAM3's default weight download hits the gated ``facebook/sam3`` Hub repo.
_GATED_HINT = (
    "SAM3 weights live in the gated Hugging Face repo 'facebook/sam3'. Request access at "
    "https://huggingface.co/facebook/sam3 and authenticate (run 'huggingface-cli login' or set "
    "HF_TOKEN), then retry. Alternatively, point ORGAN_MASKER_SAM3_CHECKPOINT at a local weights "
    "file to skip the Hub download."
)


def _is_gated_repo_error(exc: BaseException) -> bool:
    """True if ``exc`` (or its cause chain) is a Hugging Face gated-repo / auth (401) error."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if type(cur).__name__ in {"GatedRepoError", "RepositoryNotFoundError"}:
            return True
        cur = cur.__cause__ or cur.__context__
    return False


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

    def _resolve_bpe(self) -> str | None:
        """Resolve the BPE tokenizer vocab, downloading it on first use unless disabled.

        Returns an explicit path so sam3's (version-dependent) default lookup is bypassed, or
        ``None`` only when downloads are disabled and no local copy exists -- in which case sam3
        falls back to whatever vocab its install bundles (and raises its own clear error if none).
        """
        override = os.environ.get(_BPE_ENV)
        if override:
            bpe = Path(override)
            if not bpe.is_absolute():
                bpe = self._config.resolved_model_dir() / override
            if bpe.exists():
                return str(bpe)
            if not self._config.allow_download:
                raise FileNotFoundError(
                    f"SAM3 tokenizer vocab not found at {bpe} and downloads are disabled "
                    f"(--no-download); place the file there or allow downloads."
                )
        model_dir = self._config.resolved_model_dir()
        bpe = model_dir / _BPE_FILENAME
        if bpe.exists():
            return str(bpe)
        if not self._config.allow_download:
            return None
        model_dir.mkdir(parents=True, exist_ok=True)
        tmp = bpe.with_suffix(bpe.suffix + ".part")
        urllib.request.urlretrieve(_BPE_URL, tmp)  # noqa: S310 (operator-configured URL)
        tmp.replace(bpe)
        return str(bpe)

    def _get_predictor(self):
        """Build the video predictor once (weights + tokenizer vocab + device) and cache it."""
        if self._predictor is None:
            torch = self._torch
            ckpt = self._resolve_checkpoint()
            bpe = self._resolve_bpe()
            gpus = [torch.cuda.current_device()] if torch.cuda.is_available() else None
            kwargs: dict = {"gpus_to_use": gpus}
            if ckpt is not None:
                kwargs["checkpoint_path"] = ckpt
            if bpe is not None:
                kwargs["bpe_path"] = bpe
            try:
                self._predictor = self._build(**kwargs)
            except Exception as exc:  # noqa: BLE001
                if _is_gated_repo_error(exc) and ckpt is None:
                    raise RuntimeError(_GATED_HINT) from exc
                raise
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
