"""Validate OME-Zarr inputs via the ``ome-zarr-models validate`` CLI (FR-001)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


class ValidationError(RuntimeError):
    """Raised when an input fails OME-Zarr validation."""


def _find_validator() -> str | None:
    """Locate the ``ome-zarr-models`` console script (PATH, then next to the interpreter)."""
    found = shutil.which("ome-zarr-models")
    if found:
        return found
    candidate = Path(sys.executable).parent / "ome-zarr-models"
    if candidate.exists():
        return str(candidate)
    return None


def validate_ome_zarr(path: str | Path) -> None:
    """Run ``ome-zarr-models validate <path>``; raise ``ValidationError`` on failure (FR-001).

    Raises if the store is missing/invalid or if the validator CLI cannot be located.
    """
    path = Path(path)
    if not path.exists():
        raise ValidationError(f"input store does not exist: {path}")
    exe = _find_validator()
    if exe is None:
        raise ValidationError(
            "the 'ome-zarr-models' CLI is not available; install ome-zarr-models to validate inputs"
        )
    result = subprocess.run(
        [exe, "validate", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise ValidationError(f"input is not a valid OME-Zarr v0.5 store: {message}")
