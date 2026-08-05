from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from physical_ai.spatial import ADVANCED_ENERGY_METRIC, ADVANCED_MAP_METRIC


def metric_context_columns(
    contexts: np.ndarray, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    context = np.asarray(contexts, dtype=np.float64)
    if metric == "xy_ctx-patch_s4":
        raw = context[:, 103:153]
        multiplier = np.full(raw.shape[1], 4.0 / np.sqrt(raw.shape[1]))
        return raw, multiplier
    legacy_first = context[:, 103:128]
    legacy_second = context[:, 128:153]
    if metric == ADVANCED_MAP_METRIC:
        advanced = np.concatenate(
            (context[:, 153 + 54 : 153 + 58], context[:, 153 + 201 : 153 + 209]),
            axis=1,
        )
        factor = 3.0
    elif metric == ADVANCED_ENERGY_METRIC:
        advanced = context[:, 153 + 91 : 153 + 105]
        factor = 4.0
    else:
        raise ValueError(f"Unsupported local-calibration metric: {metric}")
    raw = np.concatenate((legacy_first, legacy_second, advanced), axis=1)
    multiplier = np.concatenate(
        (
            np.full(legacy_first.shape[1], 3.0 / np.sqrt(legacy_first.shape[1])),
            np.full(legacy_second.shape[1], 3.0 / np.sqrt(legacy_second.shape[1])),
            np.full(advanced.shape[1], factor / np.sqrt(advanced.shape[1])),
        )
    )
    return raw, multiplier


def fit_metric_embedding(
    positions: np.ndarray, contexts: np.ndarray, metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw, multiplier = metric_context_columns(contexts, metric)
    mean = raw.mean(axis=0)
    std = np.maximum(raw.std(axis=0), 1e-3)
    embedding = transform_metric_embedding(
        positions, contexts, metric, mean, std, multiplier
    )
    return embedding, mean, std, multiplier


def transform_metric_embedding(
    positions: np.ndarray,
    contexts: np.ndarray,
    metric: str,
    mean: np.ndarray,
    std: np.ndarray,
    multiplier: np.ndarray,
) -> np.ndarray:
    raw, expected_multiplier = metric_context_columns(contexts, metric)
    if raw.shape[1] != len(mean) or len(multiplier) != raw.shape[1]:
        raise ValueError(f"Metric normalizer width mismatch for {metric}")
    if not np.allclose(multiplier, expected_multiplier):
        raise ValueError(f"Metric multiplier mismatch for {metric}")
    xy = np.asarray(positions, dtype=np.float64)[:, :2]
    return np.concatenate((xy, (raw - mean) / std * multiplier), axis=1)


@dataclass(frozen=True)
class LocalScalarEnsemble:
    reference_residual: np.ndarray
    reference_weight: np.ndarray
    metrics: tuple[str, ...]
    reference_embeddings: tuple[np.ndarray, ...]
    context_means: tuple[np.ndarray, ...]
    context_stds: tuple[np.ndarray, ...]
    context_multipliers: tuple[np.ndarray, ...]
    neighbors: np.ndarray
    powers: np.ndarray
    softenings: np.ndarray
    energy_gammas: np.ndarray
    strengths: np.ndarray
    blend_weight: np.ndarray
    clip: float

    @classmethod
    def load(cls, path: Path) -> "LocalScalarEnsemble":
        with np.load(path) as archive:
            metrics = tuple(str(item) for item in archive["metrics"].tolist())
            count = len(metrics)
            residual_ri = np.asarray(archive["reference_residual"], dtype=np.float64)
            return cls(
                reference_residual=residual_ri[:, 0] + 1j * residual_ri[:, 1],
                reference_weight=np.asarray(archive["reference_weight"], dtype=np.float64),
                metrics=metrics,
                reference_embeddings=tuple(
                    np.asarray(archive[f"reference_embedding_{index}"], dtype=np.float64)
                    for index in range(count)
                ),
                context_means=tuple(
                    np.asarray(archive[f"context_mean_{index}"], dtype=np.float64)
                    for index in range(count)
                ),
                context_stds=tuple(
                    np.asarray(archive[f"context_std_{index}"], dtype=np.float64)
                    for index in range(count)
                ),
                context_multipliers=tuple(
                    np.asarray(archive[f"context_multiplier_{index}"], dtype=np.float64)
                    for index in range(count)
                ),
                neighbors=np.asarray(archive["neighbors"], dtype=np.int64),
                powers=np.asarray(archive["powers"], dtype=np.float64),
                softenings=np.asarray(archive["softenings"], dtype=np.float64),
                energy_gammas=np.asarray(archive["energy_gammas"], dtype=np.float64),
                strengths=np.asarray(archive["strengths"], dtype=np.float64),
                blend_weight=np.asarray(archive["blend_weight"], dtype=np.float64),
                clip=float(np.asarray(archive["clip"]).item()),
            )

    def predict(self, positions: np.ndarray, contexts: np.ndarray) -> np.ndarray:
        components = []
        for index, metric in enumerate(self.metrics):
            query = transform_metric_embedding(
                positions,
                contexts,
                metric,
                self.context_means[index],
                self.context_stds[index],
                self.context_multipliers[index],
            )
            distance, local = cKDTree(self.reference_embeddings[index]).query(
                query, k=int(self.neighbors[index]), workers=-1
            )
            weight = (distance + self.softenings[index]) ** (-self.powers[index])
            energy = self.reference_weight[local]
            energy /= np.maximum(np.median(energy, axis=1, keepdims=True), 1e-30)
            weight *= energy ** self.energy_gammas[index]
            weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-30)
            components.append(
                self.strengths[index]
                * np.sum(weight * self.reference_residual[local], axis=1)
            )
        correction = np.stack(components, axis=1) @ self.blend_weight
        magnitude = np.abs(correction)
        return correction * np.minimum(
            1.0, self.clip / np.maximum(magnitude, 1e-30)
        )
