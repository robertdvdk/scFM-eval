import numpy as np
from sklearn.metrics import r2_score


def mse(x1: np.ndarray, x2: np.ndarray) -> float:
    """Calculate Mean Squared Error."""
    return np.mean((x1 - x2) ** 2)


def wmse(x1: np.ndarray, x2: np.ndarray, weights: np.ndarray) -> float:
    """Calculate Weighted Mean Squared Error."""
    # Normalize weights to sum to 1
    normalized_weights = weights / np.sum(weights)
    return np.sum(normalized_weights * ((x1 - x2) ** 2))


def r2_score_on_deltas(delta_true: np.ndarray, delta_pred: np.ndarray, weights: np.ndarray = None) -> float:
    """Calculate R² score on delta values."""
    if len(delta_true) < 2 or len(delta_pred) < 2 or delta_true.shape != delta_pred.shape:
        return np.nan

    if weights is not None:
        return r2_score(delta_true, delta_pred, sample_weight=weights)
    else:
        return r2_score(delta_true, delta_pred)
