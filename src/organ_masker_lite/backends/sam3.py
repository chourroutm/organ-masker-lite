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
model directory) to use a specific checkpoint file instead of the Hub default.

Inference uses SAM3's request-dispatch session API (``start_session`` / ``add_prompt`` /
``propagate_in_video``). SAM3's video model is concept-driven: a bare point cannot create an
object (points only *refine* an already-tracked ``obj_id``), whereas a box alone initialises one
via the text-free detection path. organ-masker-lite's prompts are purely geometric, so for each
object we box-init (using the prompt's box, or a box synthesised around its positive points),
``propagate_in_video`` once to populate SAM3's per-frame caches, then replay the points as
``obj_id`` refinements and propagate again.

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
            return str(
                self._config.resolve_weight(override, _BPE_URL, description="SAM3 tokenizer vocab")
            )
        # No override: when downloads are disabled and no local copy exists, return None so sam3
        # falls back to whatever vocab its install bundles (rather than raising).
        if not self._config.allow_download:
            local = self._config.resolved_model_dir() / _BPE_FILENAME
            return str(local) if local.exists() else None
        return str(
            self._config.resolve_weight(_BPE_FILENAME, _BPE_URL, description="SAM3 tokenizer vocab")
        )

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

    @staticmethod
    def _synth_box_xywh(coords: np.ndarray, height: int, width: int) -> list[float]:
        """Synthesise an ``[x, y, w, h]`` box around ``coords`` (positive points).

        SAM3 cannot initialise an object from a point, so a point-only prompt is turned into a
        small box spanning its positive points plus a margin, clamped to the frame.
        """
        margin = max(2.0, 0.05 * max(height, width))
        xs, ys = coords[:, 0], coords[:, 1]
        x0 = max(0.0, float(xs.min()) - margin)
        y0 = max(0.0, float(ys.min()) - margin)
        x1 = min(float(width), float(xs.max()) + margin)
        y1 = min(float(height), float(ys.max()) + margin)
        return [x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)]

    @classmethod
    def _init_box(cls, group: list, height: int, width: int) -> tuple[int, list[float]] | None:
        """Pick a frame + normalised ``[x, y, w, h]`` init box for one object's prompts.

        Prefers an explicit box (converted from ``[x0, y0, x1, y1]``); otherwise synthesises one
        around the first prompt that carries positive points. The pixel box is normalised to the
        0--1 range SAM3's box prompt expects (``xmin/W, ymin/H, w/W, h/H``). Returns ``None`` if
        the object has neither a box nor positive points (nothing to initialise from).
        """
        box_px: list[float] | None = None
        frame_index = 0
        for p in group:
            if p.box is not None:
                x0, y0, x1, y1 = (float(v) for v in p.box)
                box_px, frame_index = [x0, y0, x1 - x0, y1 - y0], p.frame_index
                break
        if box_px is None:
            for p in group:
                pos = p.positive_coords
                if pos.size:
                    box_px, frame_index = cls._synth_box_xywh(pos, height, width), p.frame_index
                    break
        if box_px is None:
            return None
        x, y, w, h = box_px
        return frame_index, [x / width, y / height, w / width, h / height]

    def _propagate(self, predictor, session_id: str, out: np.ndarray) -> None:
        """Stream ``propagate_in_video`` and write each frame's reduced mask into ``out``."""
        height, width = out.shape[1:]
        for response in predictor.handle_stream_request(
            {"type": "propagate_in_video", "session_id": session_id}
        ):
            out[response["frame_index"]] = self._frame_mask(response["outputs"], height, width)

    def segment_video(
        self, frames: np.ndarray, prompts: PromptSet
    ) -> np.ndarray:  # pragma: no cover
        """Propagate a mask across ``frames`` using SAM3's video predictor.

        Frames are materialised to a temporary JPEG directory and a session is opened on it. Each
        object (prompts sharing an ``obj_id``) is initialised with a box, ``propagate_in_video``
        runs once to populate SAM3's per-frame caches, then the objects' points are replayed as
        refinements and propagation runs again. Each frame's ``out_binary_masks`` are reduced to a
        single boolean foreground mask.
        """
        from PIL import Image as PILImage

        torch = self._torch
        n_frames, height, width = frames.shape[:3]
        predictor = self._get_predictor()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        out = np.zeros((n_frames, height, width), dtype=bool)

        by_obj: dict[int, list] = {}
        for p in prompts.prompts:
            by_obj.setdefault(p.obj_id, []).append(p)

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
                    # Phase 1: initialise each object from a box (no text, no points).
                    for obj_id, group in by_obj.items():
                        init = self._init_box(group, height, width)
                        if init is None:
                            continue
                        frame_index, box_xywh = init
                        predictor.handle_request(
                            {
                                "type": "add_prompt",
                                "session_id": session_id,
                                "frame_index": frame_index,
                                "obj_id": obj_id,
                                "bounding_boxes": [box_xywh],
                                "bounding_box_labels": [1],
                            }
                        )
                    # Propagate once to populate the per-frame caches point refinement needs.
                    self._propagate(predictor, session_id, out)
                    # Phase 2: replay points as refinements on the now-cached frames.
                    refined = False
                    scale = np.array([width, height], dtype="float32")
                    for obj_id, group in by_obj.items():
                        for p in group:
                            if not p.point_coords.size:
                                continue
                            # SAM3's tracker points are normalised (rel_coordinates) to 0--1.
                            points = (p.point_coords.astype("float32") / scale).tolist()
                            predictor.handle_request(
                                {
                                    "type": "add_prompt",
                                    "session_id": session_id,
                                    "frame_index": p.frame_index,
                                    "obj_id": obj_id,
                                    "points": points,
                                    "point_labels": p.point_labels.astype("int32").tolist(),
                                }
                            )
                            refined = True
                    if refined:
                        out[:] = False
                        self._propagate(predictor, session_id, out)
                finally:
                    predictor.handle_request({"type": "close_session", "session_id": session_id})
        return out
