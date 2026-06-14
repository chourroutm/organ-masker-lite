# Implementation Plan: OME-Zarr Organ Masking with SAM

**Branch**: `001-ome-zarr-sam-masking` | **Date**: 2026-06-14 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-ome-zarr-sam-masking/spec.md`

## Summary

organ-masker-lite masks a 3D structure in a large OME-Zarr v0.5 volume by sweeping the volume as
stacks of 2D slices and using a SAM-family video predictor to propagate a segmentation from a few
landmark slices through each sweep, then combining the per-sweep results into one consensus mask.
The technical approach: a small pure-Python core (NumPy + Zarr v3 + ome-zarr-models) with the heavy
segmentation backends (SAM2, SAM3, PyTorch) and the interactive viewer (napari) as optional install
extras. Slices are normalized to uint8 RGB "video frames" stored in an on-disk NumPy memmap
(lossless, out-of-core); per-sweep masks accumulate into an on-disk vote memmap so peak RAM is
independent of volume depth. Prompts follow the SAM2 prompt-encoder convention (positive=1,
negative=0, box corners=2/3) so the public API mirrors SAM. The output mask is written as OME-Zarr
v0.5 with the same multiscale levels as the input, and each input is validated up front via the
`ome-zarr-models validate` CLI.

## Technical Context

**Language/Version**: Python 3.11+

**Primary Dependencies**:
- Core (minimal, required): `numpy`, `zarr>=3` (OME-Zarr v0.5 == Zarr format 3), `ome-zarr-models`
  (metadata models + the `ome-zarr-models validate` CLI), `scipy` (morphology for post-processing).
- Optional extras: `[sam2]` -> `torch` + `sam2`; `[sam3]` -> `torch` + `sam3`;
  `[interactive]` -> `napari` (+ Qt). CLI uses the stdlib `argparse` (no CLI framework dependency).

**Storage**:
- Inputs/outputs: OME-Zarr v0.5 stores on the local filesystem.
- Intermediate frames: on-disk NumPy memmap (uint8, RGB) in a temp working directory.
- Intermediate per-sweep results: on-disk NumPy memmap vote accumulator.
- Model weights: local model directory defaulting to a subdirectory of the current working
  directory (overridable via option/env), auto-downloaded on first use with an opt-out.

**Testing**: `pytest`. Synthetic small OME-Zarr fixtures generated in-repo; a fake/stub backend for
fast deterministic engine tests; real-backend tests marked and skipped when extras absent.

**Target Platform**: Linux/macOS workstation. Single CUDA GPU (>= 8 GB) recommended for the
backends; CPU fallback supported but slower. Offline operation supported with pre-placed weights.

**Project Type**: Single Python package (library + CLI + optional interactive viewer).

**Performance Goals** (Principle IV):
- Reference: a 512x512x512 volume at the selected level, single-axis forward-only, completes in
  under 5 minutes on a single CUDA GPU (>= 8 GB).
- Peak host RAM stays under 4 GB and independent of slice count along the sweep axis (out-of-core
  via memmap), so depth scales without RAM growth.
- Selecting a coarser pyramid level strictly reduces end-to-end runtime vs a finer level (SC-006).

**Constraints**:
- No eager full-volume float load; slice/chunk-wise access only.
- Output multiscale level count equals the input's level count, with input spatial metadata
  preserved so the mask registers against the input.
- Prompt representation must be byte-compatible with the SAM2 prompt encoder convention.
- Minimal required dependencies; anything heavy (torch, napari) is an optional extra.

**Scale/Scope**: Single-user, single-volume-per-run. Volumes up to multiple hundreds of GB at the
finest level (handled out-of-core by choosing a level and streaming slices). Six prioritized user
stories; this plan covers the engine and all three entry points, with US1 as the first increment.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Constitution v1.1.0 (four core principles):

- **I. Code Quality Standards** — PASS (planned). `ruff` (lint) + `ruff format` configured; typed
  public API; single shared engine behind CLI/API/interactive (FR-011) avoids duplicated logic.
- **II. Test Discipline (NON-NEGOTIABLE)** — PASS (planned). pytest from the start; a deterministic
  stub backend lets the engine, prompts, IO, and combination logic be tested without GPU/weights;
  the three named acceptance features (single point, then box, then exclusion point) are written as
  failing tests first, in that order.
- **III. Disciplined Version Control** — PASS (planned). Work decomposed into atomic, green commits
  per task; each commit builds and passes the suite.
- **IV. Performance Requirements** — PASS (planned). Explicit targets above; out-of-core memmap
  design keeps RAM bounded; a benchmark on the reference volume guards against regression.

No violations requiring justification. Two heavy dependencies (PyTorch via backend extras, SciPy
for morphology) are recorded in Complexity Tracking as justified, not gate violations.

## Project Structure

### Documentation (this feature)

```text
specs/001-ome-zarr-sam-masking/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (CLI + Python API contracts)
│   ├── cli.md
│   └── python-api.md
├── checklists/
│   └── requirements.md  # Spec quality checklist (from /speckit-specify)
└── tasks.md             # Phase 2 output (/speckit-tasks - NOT created here)
```

### Source Code (repository root)

```text
src/organ_masker_lite/
├── __init__.py
├── io/
│   ├── reader.py            # OME-Zarr v0.5 read; multiscale level access; spatial metadata
│   ├── writer.py            # OME-Zarr v0.5 write; build pyramid matching input levels
│   └── validate.py          # invoke `ome-zarr-models validate`
├── prompts/
│   └── model.py             # Prompt / PromptSet (SAM2-compatible coords, labels, box)
├── backends/
│   ├── base.py              # VideoSegmenterBackend protocol (init_state/add/propagate)
│   ├── sam2.py              # SAM2 adapter
│   ├── sam3.py              # SAM3 adapter
│   └── registry.py          # name -> backend resolution; default = sam2
├── engine/
│   ├── frames.py            # slice -> uint8 RGB frame normalization; memmap frame store
│   ├── sweep.py             # per-axis/per-direction propagation over a sweep
│   ├── combine.py           # per-voxel majority-vote consensus (memmap accumulator)
│   └── pipeline.py          # orchestration: validate -> sweeps -> combine -> postprocess -> write
├── postprocess/
│   └── morphology.py        # optional dilation/erosion/fill-holes (scipy.ndimage)
├── config.py                # RunConfig; model-directory resolution; weight download/opt-out
├── api.py                   # OrganMaskPredictor (SAM-like public API)
├── cli.py                   # argparse one-shot CLI
└── interactive.py           # napari session (optional extra; P4)

tests/
├── conftest.py              # synthetic OME-Zarr fixtures; stub backend
├── unit/                    # prompts, frames, combine, config, io
├── contract/                # CLI command schema; Python API signature/behavior vs contracts
└── integration/             # end-to-end per user story (US1 single point first)
```

**Structure Decision**: Single `src/`-layout Python package. One masking engine
(`engine/pipeline.py`) is the single source of truth invoked by `cli.py`, `api.py`, and
`interactive.py` (satisfies FR-011). Backends sit behind a `VideoSegmenterBackend` protocol so SAM2
(default) and SAM3 are interchangeable (FR-018). IO is isolated so OME-Zarr v0.5 specifics
(validation, multiscale, metadata preservation) stay in one place.

## Complexity Tracking

> Recorded per Constitution Principle I (complexity must be justified). These are accepted
> dependencies, not gate violations.

| Item | Why Needed | Simpler Alternative Rejected Because |
|------|------------|--------------------------------------|
| PyTorch + sam2/sam3 (optional extras) | The feature's entire purpose is SAM-based segmentation; the backends require torch | No lighter way to run SAM; isolating them as extras keeps the core import-light and testable without GPU |
| SciPy (core dep) | Robust dilation/erosion/fill-holes for post-processing (FR-012) | Hand-rolled morphology is error-prone and slower; SciPy's `ndimage` is the standard, well-tested implementation |
| On-disk memmap intermediates | Volumes exceed RAM; SAM2 video needs a frame sequence | Holding frames/sweeps in RAM breaks the depth-independent RAM target (Principle IV) |
