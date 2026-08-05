from __future__ import annotations

import numpy as np


def transductive_graph_features(
    labeled_positions: np.ndarray,
    unlabeled_positions: np.ndarray,
    labeled_features: np.ndarray,
    direct_prediction: np.ndarray,
    k: int = 8,
    power: float = 2.0,
    alpha: float = 0.25,
) -> np.ndarray:
    """Propagate labeled spectra through known unlabeled/test coordinates.

    The random-walk-with-restart system uses positions only; it never consumes
    test labels. ``alpha=0`` is exactly the supplied direct prediction.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    labeled_positions = np.asarray(labeled_positions)
    unlabeled_positions = np.asarray(unlabeled_positions)
    labeled_features = np.asarray(labeled_features)
    direct_prediction = np.asarray(direct_prediction)
    labeled_count = len(labeled_positions)
    unlabeled_count = len(unlabeled_positions)
    combined = np.concatenate((labeled_positions, unlabeled_positions), axis=0)
    delta = unlabeled_positions[:, None, :2] - combined[None, :, :2]
    distances = np.linalg.norm(delta, axis=-1)
    distances[np.arange(unlabeled_count), labeled_count + np.arange(unlabeled_count)] = np.inf
    neighbor = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    local_distance = np.take_along_axis(distances, neighbor, axis=1)
    weights = np.maximum(local_distance, 1e-6) ** (-power)
    weights /= weights.sum(axis=1, keepdims=True)
    transition_uu = np.zeros((unlabeled_count, unlabeled_count), dtype=np.float64)
    boundary = np.zeros_like(direct_prediction, dtype=np.float64)
    for row in range(unlabeled_count):
        labeled_mask = neighbor[row] < labeled_count
        if np.any(labeled_mask):
            boundary[row] = np.einsum(
                "k,kd->d",
                weights[row, labeled_mask],
                labeled_features[neighbor[row, labeled_mask]],
                optimize=True,
            )
        unlabeled_neighbor = neighbor[row, ~labeled_mask] - labeled_count
        transition_uu[row, unlabeled_neighbor] += weights[row, ~labeled_mask]
    right = (1.0 - alpha) * direct_prediction + alpha * boundary
    system = np.eye(unlabeled_count, dtype=np.float64) - alpha * transition_uu
    return np.linalg.solve(system, right).astype(np.float32)
