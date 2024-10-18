from typing import List
import numpy as np


# also try center = 0.2, step = 1.07, num_steps = 64
def get_quantization_points_from_geometric_grid(
    quantization_grid_center: float = 0.03,
    quantization_grid_step: float = 1.1,
    quantization_grid_num_steps: int = 64,
) -> np.ndarray:
    """Get quantization points from a geometric grid."""
    quantization_points = [
        ("%.8f" % (quantization_grid_center * quantization_grid_step**i))
        for i in range(-quantization_grid_num_steps, quantization_grid_num_steps + 1, 1)
    ]
    return np.array([float(f) for f in quantization_points])


def get_quantile_idx(quantiles: List[float], t: float) -> int:
    """Returns the quantile index that time t falls in.

    Args:
        quantiles (List[float]): List of len(quantiles)-1 quantiles where each quantile is denoted by [quantiles[i], quantiles[i+1]).
        t (float): time t that we want the quantile index of.

    Returns:
        int quantile_idx between [0, len(quantiles)-2] where t falls between quantiles[quantile_idx] and quantiles[quantile_idx+1]. If t is smaller than quantiles[0], it belongs in the first quantile. If t is greater than quantiles[-1], it belongs in the last quantile .
    """
    if t < quantiles[0]:
        return 0
    elif t > quantiles[-1]:
        return len(quantiles) - 2

    idx_to_insert_t = np.searchsorted(quantiles, t, "right")
    return idx_to_insert_t - 1
