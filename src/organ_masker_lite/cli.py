"""One-shot command-line interface (FR-008)."""

from __future__ import annotations

import argparse
import sys

from .config import COARSEST_LEVEL, RunConfig
from .prompts.model import PromptError, load_prompt_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="organ-masker-lite")
    sub = parser.add_subparsers(dest="command", required=True)

    mask = sub.add_parser("mask", help="Mask a structure in an OME-Zarr v0.5 volume.")
    mask.add_argument("input", help="Path to the input OME-Zarr v0.5 store")
    mask.add_argument("output", help="Path to the output OME-Zarr v0.5 store")
    mask.add_argument("--prompts", required=True, help="Prompt file (JSON)")
    mask.add_argument("--backend", default="sam2", help="Segmentation backend (default: sam2)")
    mask.add_argument(
        "--level",
        type=int,
        default=COARSEST_LEVEL,
        help="Multiscale level to run on (default: coarsest)",
    )
    mask.add_argument(
        "--axes",
        default="z",
        help="Comma-separated sweep axes; the first is the prompted axis (default: z)",
    )
    mask.add_argument(
        "--direction",
        default="forward",
        choices=("forward", "forward_reverse"),
        help="Propagation mode (default: forward)",
    )
    mask.add_argument(
        "--combine",
        default="majority",
        choices=("majority", "union", "intersection"),
        help="Consensus rule when combining sweeps (default: majority)",
    )
    mask.add_argument("--overwrite", action="store_true", help="Overwrite an existing output")
    mask.add_argument("--verbose", action="store_true", help="Print progress to stderr")
    mask.set_defaults(func=_cmd_mask)
    return parser


def _cmd_mask(args: argparse.Namespace) -> int:
    # Import the pipeline lazily so `--help` and arg parsing stay fast and dependency-light.
    from .engine.pipeline import PipelineError, run_masking
    from .io.reader import ReaderError
    from .io.validate import ValidationError
    from .io.writer import WriterError

    try:
        prompts = load_prompt_file(args.prompts)
    except (PromptError, FileNotFoundError, ValueError) as exc:
        print(f"error: invalid prompt file: {exc}", file=sys.stderr)
        return 1

    config = RunConfig(
        backend=args.backend,
        level=args.level,
        axes=[a.strip() for a in args.axes.split(",") if a.strip()],
        direction=args.direction,
        combine_rule=args.combine,
        overwrite=args.overwrite,
    )
    progress = (
        (lambda m: print(f"[organ-masker-lite] {m}", file=sys.stderr)) if args.verbose else None
    )

    try:
        out = run_masking(args.input, args.output, prompts, config, progress=progress)
    except (ValidationError, ReaderError, WriterError, PromptError, PipelineError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(str(out))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
