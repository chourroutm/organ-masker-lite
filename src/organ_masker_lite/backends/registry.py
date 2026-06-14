"""Backend registry: resolve a segmentation backend by name (default ``sam2``)."""

from __future__ import annotations

from collections.abc import Callable

from .base import VideoSegmenterBackend

DEFAULT_BACKEND = "sam2"

_FACTORIES: dict[str, Callable[[dict], VideoSegmenterBackend]] = {}


class BackendError(ValueError):
    """Raised when a backend name is unknown or cannot be constructed."""


def register_backend(name: str, factory: Callable[[dict], VideoSegmenterBackend]) -> None:
    """Register a backend factory under ``name`` (used by tests to inject a stub)."""
    _FACTORIES[name] = factory


def available_backends() -> list[str]:
    return sorted(_FACTORIES)


def get_backend(name: str, **options) -> VideoSegmenterBackend:
    if name not in _FACTORIES:
        known = ", ".join(available_backends()) or "(none registered)"
        raise BackendError(f"unknown backend '{name}'; available: {known}")
    return _FACTORIES[name](options)


def _make_sam2(options: dict) -> VideoSegmenterBackend:
    from .sam2 import Sam2Backend

    return Sam2Backend(**options)


register_backend("sam2", _make_sam2)
# NOTE: "sam3" is registered in its own phase (tasks T036/T037); not part of the US1 MVP.
