"""Validate OME-Zarr v0.5 inputs in-process via the installed ``ome-zarr-models`` package (FR-001).

Validation uses ``ome_zarr_models.open_ome_zarr`` directly, mirroring the package's own ``validate``
CLI (which treats :class:`ome_zarr_models.exceptions.ValidationWarning` as an error). It therefore
depends only on the installed dependency, never on a validator command being discoverable on
``PATH`` (FR-002).
"""

from __future__ import annotations

import warnings
from pathlib import Path

_MISSING_DEP_HINT = (
    "OME-Zarr validation requires the 'ome-zarr-models' package (>=1.6); "
    "install it with 'pip install ome-zarr-models'"
)


class ValidationError(RuntimeError):
    """Raised when an input fails OME-Zarr validation."""


def validate_ome_zarr(path: str | Path) -> None:
    """Validate ``path`` as an OME-Zarr v0.5 store; raise ``ValidationError`` on failure (FR-001).

    Raises ``ValidationError`` if the store is missing or invalid, or if the ``ome-zarr-models``
    dependency is unavailable.
    """
    path = Path(path)
    if not path.exists():
        raise ValidationError(f"input store does not exist: {path}")

    try:
        from ome_zarr_models import open_ome_zarr
        from ome_zarr_models.exceptions import ValidationWarning
    except ImportError as exc:  # broken/partial install (FR-005)
        raise ValidationError(_MISSING_DEP_HINT) from exc

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=ValidationWarning)
            open_ome_zarr(str(path), version="0.5")
    except Exception as exc:
        raise ValidationError(f"input is not a valid OME-Zarr v0.5 store: {exc}") from exc
