from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree


ADVANCED_MAP_METRIC = "xy_ctx-material-center-multiscale_s3"
ADVANCED_ENERGY_METRIC = "xy_ctx-endpoint-far_s4"


@dataclass(frozen=True)
class KrigingConfig:
    metric: str
    covariance: str
    neighbors: int
    bandwidth_scale: float
    nugget: float
    positive_weights: bool


def _standardized_block(raw: np.ndarray) -> np.ndarray:
    scaled = (raw - raw.mean(axis=0, keepdims=True)) / np.maximum(
        raw.std(axis=0, keepdims=True), 1e-3
    )
    return scaled / np.sqrt(scaled.shape[1])


def metric_embeddings(positions: np.ndarray, contexts: np.ndarray) -> dict[str, np.ndarray]:
    """Build validation-selected geometry and local-map embeddings."""
    xy = np.asarray(positions, dtype=np.float64)[:, :2]
    context = np.asarray(contexts, dtype=np.float64)
    result = {
        "xy_y0.75": xy * np.array([1.0, 0.75]),
    }
    for name, raw in {
        "summary": context[:, :7],
        "patch": context[:, 103:153],
        "all": context,
    }.items():
        scaled = _standardized_block(raw)
        result[f"xy_ctx-{name}_s4"] = np.concatenate((xy, 4.0 * scaled), axis=1)
    if context.shape[1] >= 362:
        # Five-fold selected advanced point-cloud metric.  The 4 center-ray
        # material channels encode wall density, normal alignment and surface
        # roughness; the final 8 channels encode smoothed height/wall fields at
        # 4/8/16/32 m.  Keep this opt-in so legacy checkpoints remain valid.
        legacy_patch = np.concatenate((context[:, 103:128], context[:, 128:153]), axis=1)
        material_multiscale = np.concatenate(
            (context[:, 153 + 54 : 153 + 58], context[:, 153 + 201 : 153 + 209]),
            axis=1,
        )
        result[ADVANCED_MAP_METRIC] = np.concatenate(
            (
                xy,
                3.0 * _standardized_block(legacy_patch[:, :25]),
                3.0 * _standardized_block(legacy_patch[:, 25:]),
                3.0 * _standardized_block(material_multiscale),
            ),
            axis=1,
        )
        # Validation-selected terminal-side point-cloud descriptor for the
        # 2x4 polarization/antenna-group energy calibration.  The final two
        # endpoint rings summarize the 16/32 m neighborhood, which is more
        # predictive of group-energy ratios than direct-path material alone.
        endpoint_far = context[:, 153 + 91 : 153 + 105]
        result[ADVANCED_ENERGY_METRIC] = np.concatenate(
            (
                xy,
                3.0 * _standardized_block(legacy_patch[:, :25]),
                3.0 * _standardized_block(legacy_patch[:, 25:]),
                4.0 * _standardized_block(endpoint_far),
            ),
            axis=1,
        )
    return result


def ordinary_kriging_weights(
    query_embedding: np.ndarray,
    neighbor_embedding: np.ndarray,
    query_distance: np.ndarray,
    bandwidth_scale: float,
    nugget: float,
    positive: bool = True,
) -> np.ndarray:
    """Return batched exponential ordinary-kriging weights."""
    del query_embedding  # distances are already relative to each query
    pair = np.linalg.norm(
        neighbor_embedding[:, :, None, :] - neighbor_embedding[:, None, :, :], axis=-1
    )
    bandwidth = np.maximum(query_distance[:, -1] * bandwidth_scale, 1e-6)
    covariance_nn = np.exp(-pair / bandwidth[:, None, None])
    covariance_query = np.exp(-query_distance / bandwidth[:, None])
    query_count, neighbor_count = query_distance.shape
    system = np.zeros((query_count, neighbor_count + 1, neighbor_count + 1), dtype=np.float64)
    system[:, :neighbor_count, :neighbor_count] = covariance_nn
    system[:, :neighbor_count, :neighbor_count] += np.eye(neighbor_count)[None] * nugget
    system[:, :neighbor_count, neighbor_count] = 1.0
    system[:, neighbor_count, :neighbor_count] = 1.0
    right = np.concatenate((covariance_query, np.ones((query_count, 1))), axis=1)
    weight = np.linalg.solve(system, right[..., None])[..., 0][:, :neighbor_count]
    if positive:
        weight = np.maximum(weight, 0.0)
        weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
    return weight.astype(np.float32)


def _covariance(distance: torch.Tensor, bandwidth: torch.Tensor, kind: str) -> torch.Tensor:
    normalized = distance / bandwidth.clamp_min(1e-6)
    if kind == "exponential":
        return torch.exp(-normalized)
    if kind == "matern32":
        scaled = np.sqrt(3.0) * normalized
        return (1.0 + scaled) * torch.exp(-scaled)
    raise ValueError(f"Unsupported covariance: {kind}")


@torch.inference_mode()
def local_ordinary_kriging(
    config: KrigingConfig,
    embedding: np.ndarray,
    train_indices: np.ndarray,
    query_indices: np.ndarray,
    features: torch.Tensor,
) -> torch.Tensor:
    """Interpolate compact spectral features with local ordinary kriging."""
    distance_np, local = cKDTree(embedding[train_indices]).query(
        embedding[query_indices], k=config.neighbors, workers=-1
    )
    neighbor_np = train_indices[local]
    neighbor_embedding = embedding[neighbor_np]
    pair_np = np.linalg.norm(
        neighbor_embedding[:, :, None, :] - neighbor_embedding[:, None, :, :], axis=-1
    ).astype(np.float32)
    device = features.device
    pair = torch.from_numpy(pair_np).to(device)
    distance = torch.from_numpy(distance_np.astype(np.float32)).to(device)
    bandwidth = distance[:, -1:, None] * config.bandwidth_scale
    covariance_nn = _covariance(pair, bandwidth, config.covariance)
    covariance_query = _covariance(distance, bandwidth[:, :, 0], config.covariance)
    query_count, neighbor_count = distance.shape
    system = torch.zeros((query_count, neighbor_count + 1, neighbor_count + 1), device=device)
    system[:, :neighbor_count, :neighbor_count] = covariance_nn
    system[:, :neighbor_count, :neighbor_count] += (
        torch.eye(neighbor_count, device=device) * config.nugget
    )
    system[:, :neighbor_count, neighbor_count] = 1.0
    system[:, neighbor_count, :neighbor_count] = 1.0
    right = torch.cat(
        (covariance_query, torch.ones((query_count, 1), device=device)), dim=1
    )
    weight = torch.linalg.solve(system, right)[:, :neighbor_count]
    if config.positive_weights:
        weight = weight.clamp_min(0.0)
        weight /= weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
    neighbor = torch.from_numpy(neighbor_np.astype(np.int64)).to(device)
    return torch.einsum("qk,qkd->qd", weight, features[neighbor]).clamp_min(0.0)


@torch.inference_mode()
def graph_propagate(
    direct: torch.Tensor,
    embedding: np.ndarray,
    train_indices: np.ndarray,
    query_indices: np.ndarray,
    features: torch.Tensor,
    neighbors: int = 24,
    power: float = 2.5,
    alpha: float = 0.1,
) -> torch.Tensor:
    """Diffuse test features over the joint train/test spatial graph."""
    combined_indices = np.concatenate((train_indices, query_indices))
    distances, local = cKDTree(embedding[combined_indices]).query(
        embedding[query_indices], k=neighbors + 1, workers=-1
    )
    distances, local = distances[:, 1:], local[:, 1:]
    weight = np.maximum(distances, 1e-8) ** (-power)
    weight /= weight.sum(axis=1, keepdims=True)
    query_count = len(query_indices)
    train_count = len(train_indices)
    transition = np.zeros((query_count, query_count), dtype=np.float32)
    boundary = torch.zeros(
        (query_count, features.shape[1]), dtype=features.dtype, device=features.device
    )
    for row in range(query_count):
        labeled = local[row] < train_count
        if np.any(labeled):
            local_weight = torch.from_numpy(weight[row, labeled].astype(np.float32)).to(features.device)
            source_index = torch.from_numpy(
                train_indices[local[row, labeled]].astype(np.int64)
            ).to(features.device)
            boundary[row] = torch.einsum("k,kd->d", local_weight, features[source_index])
        unlabeled = local[row, ~labeled] - train_count
        transition[row, unlabeled] += weight[row, ~labeled].astype(np.float32)
    transition_tensor = torch.from_numpy(transition).to(features.device)
    identity = torch.eye(query_count, device=features.device)
    matrix = identity - alpha * transition_tensor
    if direct.ndim == 2:
        return torch.linalg.solve(
            matrix, (1.0 - alpha) * direct + alpha * boundary
        ).clamp_min(0.0)
    if direct.ndim == 3:
        # Propagate a bank of candidate kernels with one shared graph solve.
        candidates, queries, width = direct.shape
        right = (1.0 - alpha) * direct + alpha * boundary[None]
        solved = torch.linalg.solve(
            matrix, right.permute(1, 0, 2).reshape(queries, candidates * width)
        )
        return solved.reshape(queries, candidates, width).permute(1, 0, 2).clamp_min(0.0)
    raise ValueError("direct must have shape (queries, features) or (candidates, queries, features)")
