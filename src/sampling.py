import numpy as np
from scipy.ndimage import binary_dilation
from config import CFG

IGNORE = CFG['ignore_index']


def sample_random(mask: np.ndarray, ratio: float = 0.01) -> np.ndarray:
    """
    Uniform random point sampling.
    Picks `ratio` fraction of pixels at random; all others get IGNORE index.
    """
    H, W   = mask.shape
    n      = max(2, int(H * W * ratio))
    flat   = mask.ravel()
    pm     = np.full(H * W, IGNORE, dtype=np.int64)
    chosen = np.random.choice(H * W, size=n, replace=False)
    pm[chosen] = flat[chosen]
    return pm.reshape(H, W)


def sample_boundary_aware(
    mask              : np.ndarray,
    ratio             : float = 0.01,
    boundary_fraction : float = 0.5,
) -> np.ndarray:
    """
    Boundary-aware point sampling.
    Half the annotation budget goes near building edges (binary dilation),
    the other half covers interior and background pixels.
    """
    H, W    = mask.shape
    n_total = max(2, int(H * W * ratio))
    flat    = mask.ravel()
    pm      = np.full(H * W, IGNORE, dtype=np.int64)

    building = (mask == 1).astype(np.uint8)
    dilated  = binary_dilation(building, iterations=3).astype(np.uint8)
    boundary = (dilated - building).ravel()

    b_idx = np.where(boundary > 0)[0]
    i_idx = np.where(boundary == 0)[0]

    n_b = min(int(n_total * boundary_fraction), len(b_idx))
    n_i = n_total - n_b

    if n_b > 0 and len(b_idx) > 0:
        sel_b     = np.random.choice(b_idx, size=n_b, replace=False)
        pm[sel_b] = flat[sel_b]

    if n_i > 0 and len(i_idx) > 0:
        sel_i     = np.random.choice(i_idx, size=min(n_i, len(i_idx)), replace=False)
        pm[sel_i] = flat[sel_i]

    return pm.reshape(H, W)


def make_point_mask(mask: np.ndarray, strategy: str = 'boundary', ratio: float = 0.01) -> np.ndarray:
    """Dispatch to the chosen sampling strategy."""
    if strategy == 'boundary':
        return sample_boundary_aware(mask, ratio)
    return sample_random(mask, ratio)
