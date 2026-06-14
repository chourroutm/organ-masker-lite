"""Optional morphological post-processing of the consensus mask (FR-012, research R8).

Conservative defaults: fill-holes on, dilation/erosion off. Operations are applied in a fixed,
documented order -- dilation, then erosion, then fill-holes -- using ``scipy.ndimage`` over the
full 3D mask. Radii are iteration counts with SciPy's default connectivity-1 structuring element.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..config import PostProcessConfig


def postprocess_mask(mask: np.ndarray, config: PostProcessConfig) -> np.ndarray:
    """Return ``mask`` with the configured morphology applied (a no-op when ``config.is_noop``)."""
    out = np.asarray(mask, dtype=bool)
    if config.is_noop:
        return out
    if config.dilation_radius > 0:
        out = ndimage.binary_dilation(out, iterations=config.dilation_radius)
    if config.erosion_radius > 0:
        out = ndimage.binary_erosion(out, iterations=config.erosion_radius)
    if config.fill_holes:
        out = ndimage.binary_fill_holes(out)
    return np.asarray(out, dtype=bool)


__all__ = ["postprocess_mask"]
