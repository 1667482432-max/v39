from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from physical_ai.expert_gate import expert_gate_condition
from physical_ai.local_calibration import transform_metric_embedding


@dataclass(frozen=True)
class _KindCorrection:
    reference_value: np.ndarray
    reference_embedding: np.ndarray
    metric: str
    context_mean: np.ndarray
    context_std: np.ndarray
    context_multiplier: np.ndarray
    neighbors: int
    power: float
    softening: float
    method: str
    strength: float
    gate_name: str
    gate_direction: str
    gate_threshold: float
    gate_second_name: str
    gate_second_direction: str
    gate_second_threshold: float

    def interpolate(self, positions: np.ndarray, contexts: np.ndarray) -> np.ndarray:
        query = transform_metric_embedding(
            positions,
            contexts,
            self.metric,
            self.context_mean,
            self.context_std,
            self.context_multiplier,
        )
        distance, local = cKDTree(self.reference_embedding).query(
            query, k=self.neighbors, workers=-1
        )
        if self.neighbors == 1:
            distance = distance[:, None]
            local = local[:, None]
        weight = (distance + self.softening) ** (-self.power)
        weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-30)
        return np.einsum(
            "qk,qkgl->qgl", weight, self.reference_value[local], optimize=True
        )


@dataclass(frozen=True)
class LocalSpectralCorrection:
    pas: _KindCorrection
    pdp: _KindCorrection
    sample_gate_coefficient: np.ndarray | None = None
    sample_gate_mean: np.ndarray | None = None
    sample_gate_std: np.ndarray | None = None
    sample_gate_fraction: float = 1.0
    sample_action_gate_coefficient: np.ndarray | None = None
    sample_action_gate_mean: np.ndarray | None = None
    sample_action_gate_std: np.ndarray | None = None
    sample_action_gate_fraction: float = 1.0
    sample_action_knn_reference: np.ndarray | None = None
    sample_action_knn_target: np.ndarray | None = None
    sample_action_knn_columns: np.ndarray | None = None
    sample_action_knn_mean: np.ndarray | None = None
    sample_action_knn_std: np.ndarray | None = None
    sample_action_knn_neighbors: int = 0
    sample_action_knn_power: float = 1.0
    sample_action_knn_softening: float = 0.0
    sample_action_knn_fraction: float = 1.0
    sample_action_physical_reference: np.ndarray | None = None
    sample_action_physical_target: np.ndarray | None = None
    sample_action_physical_metric: str = "none"
    sample_action_physical_neighbors: int = 0
    sample_action_physical_power: float = 1.0
    sample_action_physical_softening: float = 0.0
    sample_action_physical_fraction: float = 1.0
    sample_action_physical_bias: np.ndarray | None = None
    sample_action_physical_transform: np.ndarray | None = None
    sample_action_physical_secondary_reference: np.ndarray | None = None
    sample_action_physical_secondary_target: np.ndarray | None = None
    sample_action_physical_secondary_metric: str = "none"
    sample_action_physical_secondary_neighbors: int = 0
    sample_action_physical_secondary_power: float = 1.0
    sample_action_physical_secondary_softening: float = 0.0
    sample_action_physical_secondary_transform: np.ndarray | None = None
    sample_action_physical_ensemble_weight: float = 1.0

    @classmethod
    def load(cls, path: Path) -> "LocalSpectralCorrection":
        with np.load(path) as archive:
            def load_kind(kind: str) -> _KindCorrection:
                return _KindCorrection(
                    reference_value=np.asarray(
                        archive[f"{kind}_reference_value"], dtype=np.float32
                    ),
                    reference_embedding=np.asarray(
                        archive[f"{kind}_reference_embedding"], dtype=np.float64
                    ),
                    metric=str(np.asarray(archive[f"{kind}_metric"]).item()),
                    context_mean=np.asarray(
                        archive[f"{kind}_context_mean"], dtype=np.float64
                    ),
                    context_std=np.asarray(
                        archive[f"{kind}_context_std"], dtype=np.float64
                    ),
                    context_multiplier=np.asarray(
                        archive[f"{kind}_context_multiplier"], dtype=np.float64
                    ),
                    neighbors=int(np.asarray(archive[f"{kind}_neighbors"]).item()),
                    power=float(np.asarray(archive[f"{kind}_power"]).item()),
                    softening=float(np.asarray(archive[f"{kind}_softening"]).item()),
                    method=str(np.asarray(archive[f"{kind}_method"]).item()),
                    strength=float(np.asarray(archive[f"{kind}_strength"]).item()),
                    gate_name=str(
                        np.asarray(
                            archive.get(f"{kind}_gate_name", np.array("always"))
                        ).item()
                    ),
                    gate_direction=str(
                        np.asarray(
                            archive.get(f"{kind}_gate_direction", np.array("all"))
                        ).item()
                    ),
                    gate_threshold=float(
                        np.asarray(
                            archive.get(f"{kind}_gate_threshold", np.array(0.0))
                        ).item()
                    ),
                    gate_second_name=str(
                        np.asarray(
                            archive.get(f"{kind}_gate_second_name", np.array("none"))
                        ).item()
                    ),
                    gate_second_direction=str(
                        np.asarray(
                            archive.get(
                                f"{kind}_gate_second_direction", np.array("none")
                            )
                        ).item()
                    ),
                    gate_second_threshold=float(
                        np.asarray(
                            archive.get(
                                f"{kind}_gate_second_threshold", np.array(0.0)
                            )
                        ).item()
                    ),
                )

            coefficient = (
                np.asarray(archive["sample_gate_coefficient"], dtype=np.float32)
                if "sample_gate_coefficient" in archive
                else None
            )
            mean = (
                np.asarray(archive["sample_gate_mean"], dtype=np.float32)
                if "sample_gate_mean" in archive
                else None
            )
            std = (
                np.asarray(archive["sample_gate_std"], dtype=np.float32)
                if "sample_gate_std" in archive
                else None
            )
            fraction = float(
                np.asarray(archive.get("sample_gate_fraction", np.array(1.0))).item()
            )
            action_coefficient = (
                np.asarray(
                    archive["sample_action_gate_coefficient"], dtype=np.float32
                )
                if "sample_action_gate_coefficient" in archive
                else None
            )
            action_mean = (
                np.asarray(archive["sample_action_gate_mean"], dtype=np.float32)
                if "sample_action_gate_mean" in archive
                else None
            )
            action_std = (
                np.asarray(archive["sample_action_gate_std"], dtype=np.float32)
                if "sample_action_gate_std" in archive
                else None
            )
            action_fraction = float(
                np.asarray(
                    archive.get("sample_action_gate_fraction", np.array(1.0))
                ).item()
            )
            knn_reference = (
                np.asarray(archive["sample_action_knn_reference"], dtype=np.float32)
                if "sample_action_knn_reference" in archive
                else None
            )
            knn_target = (
                np.asarray(archive["sample_action_knn_target"], dtype=np.float32)
                if "sample_action_knn_target" in archive
                else None
            )
            knn_columns = (
                np.asarray(archive["sample_action_knn_columns"], dtype=np.int64)
                if "sample_action_knn_columns" in archive
                else None
            )
            knn_mean = (
                np.asarray(archive["sample_action_knn_mean"], dtype=np.float32)
                if "sample_action_knn_mean" in archive
                else None
            )
            knn_std = (
                np.asarray(archive["sample_action_knn_std"], dtype=np.float32)
                if "sample_action_knn_std" in archive
                else None
            )
            physical_reference = (
                np.asarray(
                    archive["sample_action_physical_reference"], dtype=np.float64
                )
                if "sample_action_physical_reference" in archive
                else None
            )
            physical_target = (
                np.asarray(
                    archive["sample_action_physical_target"], dtype=np.float32
                )
                if "sample_action_physical_target" in archive
                else None
            )
            return cls(
                pas=load_kind("pas"),
                pdp=load_kind("pdp"),
                sample_gate_coefficient=coefficient,
                sample_gate_mean=mean,
                sample_gate_std=std,
                sample_gate_fraction=fraction,
                sample_action_gate_coefficient=action_coefficient,
                sample_action_gate_mean=action_mean,
                sample_action_gate_std=action_std,
                sample_action_gate_fraction=action_fraction,
                sample_action_knn_reference=knn_reference,
                sample_action_knn_target=knn_target,
                sample_action_knn_columns=knn_columns,
                sample_action_knn_mean=knn_mean,
                sample_action_knn_std=knn_std,
                sample_action_knn_neighbors=int(
                    np.asarray(
                        archive.get("sample_action_knn_neighbors", np.array(0))
                    ).item()
                ),
                sample_action_knn_power=float(
                    np.asarray(
                        archive.get("sample_action_knn_power", np.array(1.0))
                    ).item()
                ),
                sample_action_knn_softening=float(
                    np.asarray(
                        archive.get("sample_action_knn_softening", np.array(0.0))
                    ).item()
                ),
                sample_action_knn_fraction=float(
                    np.asarray(
                        archive.get("sample_action_knn_fraction", np.array(1.0))
                    ).item()
                ),
                sample_action_physical_reference=physical_reference,
                sample_action_physical_target=physical_target,
                sample_action_physical_metric=str(
                    np.asarray(
                        archive.get("sample_action_physical_metric", np.array("none"))
                    ).item()
                ),
                sample_action_physical_neighbors=int(
                    np.asarray(
                        archive.get("sample_action_physical_neighbors", np.array(0))
                    ).item()
                ),
                sample_action_physical_power=float(
                    np.asarray(
                        archive.get("sample_action_physical_power", np.array(1.0))
                    ).item()
                ),
                sample_action_physical_softening=float(
                    np.asarray(
                        archive.get("sample_action_physical_softening", np.array(0.0))
                    ).item()
                ),
                sample_action_physical_fraction=float(
                    np.asarray(
                        archive.get("sample_action_physical_fraction", np.array(1.0))
                    ).item()
                ),
                sample_action_physical_bias=(
                    np.asarray(
                        archive["sample_action_physical_bias"], dtype=np.float64
                    )
                    if "sample_action_physical_bias" in archive
                    else None
                ),
                sample_action_physical_transform=(
                    np.asarray(
                        archive["sample_action_physical_transform"], dtype=np.float64
                    )
                    if "sample_action_physical_transform" in archive
                    else None
                ),
                sample_action_physical_secondary_reference=(
                    np.asarray(
                        archive["sample_action_physical_secondary_reference"],
                        dtype=np.float64,
                    )
                    if "sample_action_physical_secondary_reference" in archive
                    else None
                ),
                sample_action_physical_secondary_target=(
                    np.asarray(
                        archive["sample_action_physical_secondary_target"],
                        dtype=np.float32,
                    )
                    if "sample_action_physical_secondary_target" in archive
                    else None
                ),
                sample_action_physical_secondary_metric=str(
                    np.asarray(
                        archive.get(
                            "sample_action_physical_secondary_metric", np.array("none")
                        )
                    ).item()
                ),
                sample_action_physical_secondary_neighbors=int(
                    np.asarray(
                        archive.get(
                            "sample_action_physical_secondary_neighbors", np.array(0)
                        )
                    ).item()
                ),
                sample_action_physical_secondary_power=float(
                    np.asarray(
                        archive.get(
                            "sample_action_physical_secondary_power", np.array(1.0)
                        )
                    ).item()
                ),
                sample_action_physical_secondary_softening=float(
                    np.asarray(
                        archive.get(
                            "sample_action_physical_secondary_softening", np.array(0.0)
                        )
                    ).item()
                ),
                sample_action_physical_secondary_transform=(
                    np.asarray(
                        archive["sample_action_physical_secondary_transform"],
                        dtype=np.float64,
                    )
                    if "sample_action_physical_secondary_transform" in archive
                    else None
                ),
                sample_action_physical_ensemble_weight=float(
                    np.asarray(
                        archive.get(
                            "sample_action_physical_ensemble_weight", np.array(1.0)
                        )
                    ).item()
                ),
            )

    @staticmethod
    def _group_state(
        prediction: torch.Tensor,
        local_value: np.ndarray,
        correction: _KindCorrection,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        norm = torch.linalg.vector_norm(prediction, dim=-1, keepdim=True).clamp_min(
            1e-30
        )
        base = prediction / norm
        local = torch.from_numpy(local_value).to(
            device=prediction.device, dtype=prediction.dtype
        )
        consensus = torch.linalg.vector_norm(local, dim=-1)
        local_unit = local / consensus.unsqueeze(-1).clamp_min(1e-30)
        features = {
            "agreement": torch.sum(base * local_unit, dim=-1),
            "consensus": consensus,
            "delta": torch.linalg.vector_norm(local - base, dim=-1),
        }

        def compare(name: str, direction: str, threshold: float) -> torch.Tensor:
            if name not in features:
                raise ValueError(f"Unsupported spectral gate feature: {name}")
            if direction == "low":
                return features[name] <= threshold
            if direction == "high":
                return features[name] >= threshold
            raise ValueError(f"Unsupported spectral gate direction: {direction}")

        if correction.gate_name == "always":
            gate = torch.ones_like(consensus, dtype=torch.bool)
        elif correction.gate_name == "consensus_agreement":
            gate = compare("consensus", "high", correction.gate_threshold)
            gate &= compare(
                correction.gate_second_name,
                correction.gate_second_direction,
                correction.gate_second_threshold,
            )
        else:
            gate = compare(
                correction.gate_name,
                correction.gate_direction,
                correction.gate_threshold,
            )
        return base, local, features, gate

    @staticmethod
    def _summarize_group_state(
        features: dict[str, torch.Tensor], gate: torch.Tensor
    ) -> torch.Tensor:
        rows = []
        for name in ("agreement", "consensus", "delta"):
            value = features[name]
            rows.extend(
                (
                    value.mean(dim=1),
                    value.std(dim=1, unbiased=False),
                    value.amin(dim=1),
                    value.amax(dim=1),
                    torch.quantile(value, 0.25, dim=1),
                    torch.quantile(value, 0.75, dim=1),
                )
            )
        gate_float = gate.to(features["delta"].dtype)
        rows.extend(
            (
                gate_float.mean(dim=1),
                (gate_float * features["delta"]).mean(dim=1),
                (gate_float * features["delta"]).amax(dim=1),
            )
        )
        return torch.stack(rows, dim=1)

    def _sample_feature_tensor(
        self,
        pas: torch.Tensor,
        pdp: torch.Tensor,
        pas_local: np.ndarray,
        pdp_local: np.ndarray,
        positions: np.ndarray,
        contexts: np.ndarray,
    ) -> torch.Tensor:
        _, _, pas_features, pas_gate = self._group_state(
            pas, pas_local, self.pas
        )
        _, _, pdp_features, pdp_gate = self._group_state(
            pdp, pdp_local, self.pdp
        )
        condition = torch.from_numpy(
            expert_gate_condition(positions, contexts, "advanced")
        ).to(device=pas.device, dtype=pas.dtype)
        return torch.cat(
            (
                condition,
                self._summarize_group_state(pas_features, pas_gate),
                self._summarize_group_state(pdp_features, pdp_gate),
            ),
            dim=1,
        )

    def sample_features(
        self,
        compact: torch.Tensor,
        positions: np.ndarray,
        contexts: np.ndarray,
    ) -> np.ndarray:
        pas = compact[:, :1024].reshape(-1, 256, 4).permute(0, 2, 1)
        pdp = compact[:, 1024:].reshape(-1, 2, 4, 192).reshape(-1, 8, 192)
        pas_local = self.pas.interpolate(positions, contexts)
        pdp_local = self.pdp.interpolate(positions, contexts)
        return (
            self._sample_feature_tensor(
                pas, pdp, pas_local, pdp_local, positions, contexts
            )
            .detach()
            .cpu()
            .numpy()
        )

    @staticmethod
    def _apply(
        prediction: torch.Tensor,
        local_value: np.ndarray,
        correction: _KindCorrection,
        sample_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm = torch.linalg.vector_norm(prediction, dim=-1, keepdim=True).clamp_min(
            1e-30
        )
        base, local, _, gate = LocalSpectralCorrection._group_state(
            prediction, local_value, correction
        )
        strength = correction.strength * gate.unsqueeze(-1)
        if sample_gate is not None:
            strength = strength * sample_gate[:, None, None]
        if correction.method == "residual":
            corrected = base + strength * local
        elif correction.method == "target":
            corrected = base + strength * (local - base)
        else:
            raise ValueError(f"Unsupported spectral correction method: {correction.method}")
        corrected = corrected.clamp_min(0.0)
        corrected /= torch.linalg.vector_norm(
            corrected, dim=-1, keepdim=True
        ).clamp_min(1e-30)
        return corrected * norm

    @staticmethod
    def _physical_action_value(
        positions: np.ndarray,
        reference: np.ndarray,
        target: np.ndarray,
        metric: str,
        transform: np.ndarray | None,
        neighbors: int,
        power: float,
        softening: float,
    ) -> np.ndarray:
        metric_prefix = "xy_y"
        if metric.startswith(metric_prefix):
            try:
                y_scale = float(metric[len(metric_prefix) :])
            except ValueError as exc:
                raise ValueError(f"Unsupported physical action metric: {metric}") from exc
            if not np.isfinite(y_scale) or y_scale <= 0.0:
                raise ValueError(f"Unsupported physical action metric: {metric}")
            query = np.asarray(positions, dtype=np.float64)[:, :2] * np.array(
                [1.0, y_scale]
            )
        elif metric == "xy_matrix":
            if transform is None or transform.shape != (2, 2):
                raise ValueError("Physical action transform must be a 2x2 matrix")
            query = np.asarray(positions, dtype=np.float64)[:, :2] @ transform
        else:
            raise ValueError(f"Unsupported physical action metric: {metric}")
        distance, local = cKDTree(reference).query(
            query, k=neighbors, workers=-1
        )
        if neighbors == 1:
            distance, local = distance[:, None], local[:, None]
        weight = (distance + softening) ** (-power)
        weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-30)
        return np.einsum(
            "qk,qka->qa", weight, target[local], optimize=True
        )

    def apply(
        self,
        compact: torch.Tensor,
        positions: np.ndarray,
        contexts: np.ndarray,
    ) -> torch.Tensor:
        pas = compact[:, :1024].reshape(-1, 256, 4).permute(0, 2, 1)
        pdp = compact[:, 1024:].reshape(-1, 2, 4, 192).reshape(-1, 8, 192)
        pas_local = self.pas.interpolate(positions, contexts)
        pdp_local = self.pdp.interpolate(positions, contexts)
        pas_sample_gate = None
        pdp_sample_gate = None
        if self.sample_action_physical_reference is not None:
            if self.sample_action_physical_target is None:
                raise ValueError("Incomplete physical four-action gate calibration")
            action_value = self._physical_action_value(
                positions,
                self.sample_action_physical_reference,
                self.sample_action_physical_target,
                self.sample_action_physical_metric,
                self.sample_action_physical_transform,
                self.sample_action_physical_neighbors,
                self.sample_action_physical_power,
                self.sample_action_physical_softening,
            )
            if self.sample_action_physical_secondary_reference is not None:
                if self.sample_action_physical_secondary_target is None:
                    raise ValueError(
                        "Incomplete secondary physical four-action gate calibration"
                    )
                blend = self.sample_action_physical_ensemble_weight
                if not 0.0 <= blend <= 1.0:
                    raise ValueError("Physical action ensemble weight must be in [0, 1]")
                secondary_value = self._physical_action_value(
                    positions,
                    self.sample_action_physical_secondary_reference,
                    self.sample_action_physical_secondary_target,
                    self.sample_action_physical_secondary_metric,
                    self.sample_action_physical_secondary_transform,
                    self.sample_action_physical_secondary_neighbors,
                    self.sample_action_physical_secondary_power,
                    self.sample_action_physical_secondary_softening,
                )
                action_value = blend * action_value + (1.0 - blend) * secondary_value
            if self.sample_action_physical_bias is not None:
                if self.sample_action_physical_bias.shape != (3,):
                    raise ValueError("Physical action bias must contain three values")
                action_value = action_value + self.sample_action_physical_bias
            value = torch.from_numpy(action_value).to(
                device=compact.device, dtype=compact.dtype
            )
            benefit, action = torch.max(value, dim=1)
            count = int(round(self.sample_action_physical_fraction * len(value)))
            count = min(max(count, 0), len(value))
            selected = torch.zeros_like(benefit, dtype=torch.bool)
            if count > 0:
                chosen = torch.topk(benefit, count, sorted=False).indices
                selected[chosen] = True
            action = action + 1
            pas_sample_gate = (
                selected & ((action == 1) | (action == 3))
            ).to(compact.dtype)
            pdp_sample_gate = (
                selected & ((action == 2) | (action == 3))
            ).to(compact.dtype)
        elif self.sample_action_knn_reference is not None:
            if (
                self.sample_action_knn_target is None
                or self.sample_action_knn_columns is None
                or self.sample_action_knn_mean is None
                or self.sample_action_knn_std is None
            ):
                raise ValueError("Incomplete KNN four-action spectral gate calibration")
            raw = self._sample_feature_tensor(
                pas, pdp, pas_local, pdp_local, positions, contexts
            ).detach().cpu().numpy()
            columns = self.sample_action_knn_columns
            query = (
                raw[:, columns] - self.sample_action_knn_mean[columns]
            ) / np.maximum(self.sample_action_knn_std[columns], 1e-6)
            distance, local = cKDTree(self.sample_action_knn_reference).query(
                query, k=self.sample_action_knn_neighbors, workers=-1
            )
            if self.sample_action_knn_neighbors == 1:
                distance, local = distance[:, None], local[:, None]
            weight = (distance + self.sample_action_knn_softening) ** (
                -self.sample_action_knn_power
            )
            weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-30)
            action_value = np.einsum(
                "qk,qka->qa",
                weight,
                self.sample_action_knn_target[local],
                optimize=True,
            )
            value = torch.from_numpy(action_value).to(
                device=compact.device, dtype=compact.dtype
            )
            benefit, action = torch.max(value, dim=1)
            count = int(round(self.sample_action_knn_fraction * len(value)))
            count = min(max(count, 0), len(value))
            selected = torch.zeros_like(benefit, dtype=torch.bool)
            if count > 0:
                chosen = torch.topk(benefit, count, sorted=False).indices
                selected[chosen] = True
            action = action + 1
            pas_sample_gate = (
                selected & ((action == 1) | (action == 3))
            ).to(compact.dtype)
            pdp_sample_gate = (
                selected & ((action == 2) | (action == 3))
            ).to(compact.dtype)
        elif self.sample_action_gate_coefficient is not None:
            if self.sample_action_gate_mean is None or self.sample_action_gate_std is None:
                raise ValueError("Incomplete four-action spectral gate calibration")
            raw = self._sample_feature_tensor(
                pas, pdp, pas_local, pdp_local, positions, contexts
            )
            mean = torch.from_numpy(self.sample_action_gate_mean).to(
                device=compact.device, dtype=compact.dtype
            )
            std = torch.from_numpy(self.sample_action_gate_std).to(
                device=compact.device, dtype=compact.dtype
            )
            coefficient = torch.from_numpy(
                self.sample_action_gate_coefficient
            ).to(device=compact.device, dtype=compact.dtype)
            value = ((raw - mean) / std.clamp_min(1e-6)) @ coefficient[:, :-1].T
            value = value + coefficient[:, -1]
            benefit, action = torch.max(value, dim=1)
            count = int(round(self.sample_action_gate_fraction * len(value)))
            count = min(max(count, 0), len(value))
            selected = torch.zeros_like(benefit, dtype=torch.bool)
            if count > 0:
                chosen = torch.topk(benefit, count, sorted=False).indices
                selected[chosen] = True
            action = action + 1
            pas_sample_gate = (
                selected & ((action == 1) | (action == 3))
            ).to(compact.dtype)
            pdp_sample_gate = (
                selected & ((action == 2) | (action == 3))
            ).to(compact.dtype)
        elif self.sample_gate_coefficient is not None:
            if self.sample_gate_mean is None or self.sample_gate_std is None:
                raise ValueError("Incomplete sample-level spectral gate calibration")
            raw = self._sample_feature_tensor(
                pas, pdp, pas_local, pdp_local, positions, contexts
            )
            mean = torch.from_numpy(self.sample_gate_mean).to(
                device=compact.device, dtype=compact.dtype
            )
            std = torch.from_numpy(self.sample_gate_std).to(
                device=compact.device, dtype=compact.dtype
            )
            coefficient = torch.from_numpy(self.sample_gate_coefficient).to(
                device=compact.device, dtype=compact.dtype
            )
            value = ((raw - mean) / std.clamp_min(1e-6)) @ coefficient[:-1]
            value = value + coefficient[-1]
            count = int(round(self.sample_gate_fraction * len(value)))
            count = min(max(count, 0), len(value))
            sample_gate = torch.zeros_like(value)
            if count > 0:
                selected = torch.topk(value, count, sorted=False).indices
                sample_gate[selected] = 1.0
            pas_sample_gate = sample_gate
            pdp_sample_gate = sample_gate
        pas = self._apply(pas, pas_local, self.pas, pas_sample_gate)
        pdp = self._apply(pdp, pdp_local, self.pdp, pdp_sample_gate)
        return torch.cat(
            (
                pas.permute(0, 2, 1).reshape(len(compact), -1),
                pdp.reshape(len(compact), -1),
            ),
            dim=1,
        )
