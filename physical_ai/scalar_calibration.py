from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from physical_ai.expert_gate import expert_gate_condition


def scalar_calibration_features(
    positions: np.ndarray,
    contexts: np.ndarray,
    nearest_distance: np.ndarray,
    final_pred_energy: np.ndarray,
    pred_energy_pol_ue: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Build observable features for per-sample complex channel calibration."""
    position = np.asarray(positions, dtype=np.float32)
    context = np.asarray(contexts, dtype=np.float32)
    condition_mode = "basic" if mode == "basic" else "advanced"
    condition = expert_gate_condition(position, context, condition_mode)
    delta = position - np.array([50.0, 0.0, 25.0], dtype=np.float32)
    radius = np.linalg.norm(delta, axis=1, keepdims=True)
    direction = delta / np.maximum(radius, 1e-6)
    group_energy = np.asarray(pred_energy_pol_ue, dtype=np.float64).reshape(
        len(position), -1
    )
    group_fraction = group_energy / np.maximum(
        group_energy.sum(1, keepdims=True), 1e-30
    )
    observable = np.concatenate(
        (
            np.asarray(nearest_distance, dtype=np.float64).reshape(-1, 1),
            np.log(
                np.asarray(final_pred_energy, dtype=np.float64)
                .reshape(-1, 1)
                .clip(1e-30)
            ),
            np.log(group_fraction.clip(1e-12)),
            radius,
            direction,
        ),
        axis=1,
    )
    value = np.concatenate((condition, observable), axis=1).astype(np.float64)
    if mode != "advanced_rbf":
        return value
    x_centers = np.linspace(50.0, 250.0, 9)
    y_centers = np.linspace(-200.0, 100.0, 9)
    centers = np.stack(np.meshgrid(x_centers, y_centers), axis=-1).reshape(-1, 2)
    squared = np.sum((position[:, None, :2] - centers[None]) ** 2, axis=2)
    rbf = np.concatenate(
        [
            np.exp(-squared / (2.0 * scale * scale))
            for scale in (20.0, 40.0, 80.0)
        ],
        axis=1,
    )
    return np.concatenate((value, rbf), axis=1)


def weighted_standardize(
    features: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = weight / np.maximum(weight.sum(), 1e-30)
    mean = np.sum(normalized[:, None] * features, axis=0)
    variance = np.sum(normalized[:, None] * (features - mean) ** 2, axis=0)
    std = np.sqrt(np.maximum(variance, 1e-8))
    return (features - mean) / std, mean, std


def fit_weighted_ridge(
    features: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    standardized, mean, std = weighted_standardize(features, weight)
    design = np.concatenate((np.ones((len(features), 1)), standardized), axis=1)
    normalized_weight = weight / np.maximum(weight.mean(), 1e-30)
    gram = design.T @ (normalized_weight[:, None] * design)
    penalty = np.eye(design.shape[1]) * regularization
    penalty[0, 0] = 0.0
    right = design.T @ (normalized_weight[:, None] * target)
    coefficient = np.linalg.solve(gram + penalty, right)
    return coefficient, mean, std


def ridge_prediction(
    features: np.ndarray,
    coefficient: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    design = np.concatenate(
        (np.ones((len(features), 1)), (features - mean) / std), axis=1
    )
    return design @ coefficient


@dataclass(frozen=True)
class ScalarCalibration:
    coefficient: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    global_scale: complex
    strength: float
    mode: str

    @classmethod
    def load(cls, path: Path) -> "ScalarCalibration":
        with np.load(path) as archive:
            global_scale = np.asarray(archive["global_scale"], dtype=np.float64)
            return cls(
                coefficient=np.asarray(archive["coefficient"], dtype=np.float64),
                feature_mean=np.asarray(archive["feature_mean"], dtype=np.float64),
                feature_std=np.asarray(archive["feature_std"], dtype=np.float64),
                global_scale=complex(global_scale[0], global_scale[1]),
                strength=float(np.asarray(archive["strength"]).item()),
                mode=str(np.asarray(archive["mode"]).item()),
            )

    def predict(
        self,
        positions: np.ndarray,
        contexts: np.ndarray,
        nearest_distance: np.ndarray,
        final_pred_energy: np.ndarray,
        pred_energy_pol_ue: np.ndarray,
    ) -> np.ndarray:
        features = scalar_calibration_features(
            positions,
            contexts,
            nearest_distance,
            final_pred_energy,
            pred_energy_pol_ue,
            self.mode,
        )
        raw = ridge_prediction(
            features, self.coefficient, self.feature_mean, self.feature_std
        )
        predicted = raw[:, 0] + 1j * raw[:, 1]
        return self.global_scale + self.strength * (predicted - self.global_scale)


@dataclass(frozen=True)
class UECalibrationResidual:
    coefficient: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    strength: float
    mode: str

    @classmethod
    def load(cls, path: Path) -> "UECalibrationResidual":
        with np.load(path) as archive:
            groups = int(np.asarray(archive["groups"]).item())
            if groups != 4:
                raise ValueError(f"Expected four UE calibration groups, got {groups}")
            return cls(
                coefficient=np.asarray(archive["coefficient"], dtype=np.float64),
                feature_mean=np.asarray(archive["feature_mean"], dtype=np.float64),
                feature_std=np.asarray(archive["feature_std"], dtype=np.float64),
                strength=float(np.asarray(archive["strength"]).item()),
                mode=str(np.asarray(archive["mode"]).item()),
            )

    def predict(
        self,
        positions: np.ndarray,
        contexts: np.ndarray,
        nearest_distance: np.ndarray,
        final_pred_energy: np.ndarray,
        pred_energy_pol_ue: np.ndarray,
    ) -> np.ndarray:
        features = scalar_calibration_features(
            positions,
            contexts,
            nearest_distance,
            final_pred_energy,
            pred_energy_pol_ue,
            self.mode,
        )
        values = [
            ridge_prediction(
                features,
                self.coefficient[ue],
                self.feature_mean[ue],
                self.feature_std[ue],
            )
            for ue in range(len(self.coefficient))
        ]
        real_imag = np.stack(values, axis=1)
        return self.strength * (real_imag[..., 0] + 1j * real_imag[..., 1])
