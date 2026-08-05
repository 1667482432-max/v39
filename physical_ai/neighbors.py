from __future__ import annotations

import numpy as np


def nearest_neighbors(
    query_positions: np.ndarray,
    reference_positions: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 1 <= k <= len(reference_positions):
        raise ValueError("k must be between 1 and the reference-set size")
    delta = np.asarray(query_positions, dtype=np.float64)[:, None, :] - np.asarray(
        reference_positions, dtype=np.float64
    )[None, :, :]
    distances_squared = np.einsum("qrd,qrd->qr", delta, delta)
    unordered = np.argpartition(distances_squared, kth=k - 1, axis=1)[:, :k]
    local_d2 = np.take_along_axis(distances_squared, unordered, axis=1)
    order = np.argsort(local_d2, axis=1)
    indices = np.take_along_axis(unordered, order, axis=1)
    distances = np.sqrt(np.take_along_axis(local_d2, order, axis=1))
    return indices, distances


def distance_weights(
    distances: np.ndarray,
    power: float = 2.0,
    bandwidth: float | None = None,
    epsilon: float = 1e-6,
) -> np.ndarray:
    if bandwidth is None:
        weights = np.power(np.maximum(distances, epsilon), -power)
    else:
        weights = np.exp(-0.5 * np.square(distances / bandwidth))
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), epsilon)


def interpolate_complex(
    channels: np.ndarray,
    neighbor_indices: np.ndarray,
    weights: np.ndarray,
    query_index: int,
) -> np.ndarray:
    selected = np.asarray(channels[neighbor_indices[query_index]], dtype=np.complex64)
    local_weights = weights[query_index].astype(np.float32)
    return np.einsum("k,k...->...", local_weights, selected, optimize=True)


def delaunay_neighbors(
    query_positions: np.ndarray,
    reference_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return triangle vertices and barycentric weights for 2-D queries.

    Queries outside the reference convex hull are marked by ``inside=False`` and
    receive zero placeholders so callers can apply a KNN fallback.
    """
    from scipy.spatial import Delaunay

    reference_xy = np.asarray(reference_positions, dtype=np.float64)[:, :2]
    query_xy = np.asarray(query_positions, dtype=np.float64)[:, :2]
    triangulation = Delaunay(reference_xy)
    simplex = triangulation.find_simplex(query_xy)
    inside = simplex >= 0
    safe_simplex = np.maximum(simplex, 0)
    transform = triangulation.transform[safe_simplex]
    first_two = np.einsum("qij,qj->qi", transform[:, :2], query_xy - transform[:, 2])
    weights = np.column_stack((first_two, 1.0 - first_two.sum(axis=1)))
    indices = triangulation.simplices[safe_simplex]
    weights[~inside] = 0.0
    indices[~inside] = 0
    return indices, weights, inside


def affine_reproduction_weights(
    query_positions: np.ndarray,
    neighbor_positions: np.ndarray,
    base_weights: np.ndarray,
    ridge: float = 1e-5,
) -> np.ndarray:
    """Local-linear interpolation weights that reproduce an affine field."""
    queries = np.asarray(query_positions, dtype=np.float64)
    neighbors = np.asarray(neighbor_positions, dtype=np.float64)
    base = np.asarray(base_weights, dtype=np.float64)
    output = np.empty_like(base)
    for i in range(len(queries)):
        delta = neighbors[i, :, :2] - queries[i, None, :2]
        design = np.column_stack((np.ones(len(delta)), delta))
        weighted_gram = design.T @ (base[i, :, None] * design)
        regularizer = np.diag([ridge * 1e-3, ridge, ridge])
        solve = np.linalg.solve(weighted_gram + regularizer, np.array([1.0, 0.0, 0.0]))
        output[i] = base[i] * (design @ solve)
        output[i] /= max(output[i].sum(), 1e-12)
    return output
