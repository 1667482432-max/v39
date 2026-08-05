from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from physical_ai.data import RoundData
from physical_ai.features import nonzero_feature_indices, spectral_targets_from_features
from physical_ai.expert_gate import ExpertDisagreementGate, expert_gate_condition
from physical_ai.model import MapConditionedKernel, interpolate_features
from physical_ai.neighbors import nearest_neighbors
from physical_ai.local_calibration import LocalScalarEnsemble
from physical_ai.scalar_calibration import ScalarCalibration, UECalibrationResidual
from physical_ai.spectral_calibration import LocalSpectralCorrection
from physical_ai.spatial import (
    ADVANCED_MAP_METRIC,
    KrigingConfig,
    graph_propagate,
    local_ordinary_kriging,
    metric_embeddings,
    ordinary_kriging_weights,
)
from physical_ai.spectral import replace_fourier_magnitude


DEFAULT_MODELS = tuple(
    Path(f"artifacts/final_model_seed{seed}.pt")
    for seed in (7, 19, 43, 101, 20260804)
)
GROUP_CONFIGS = (
    KrigingConfig("xy_ctx-patch_s4", "exponential", 16, 0.5, 0.001, False),
    KrigingConfig("xy_ctx-patch_s4", "exponential", 16, 0.5, 0.01, False),
    KrigingConfig("xy_ctx-patch_s4", "exponential", 16, 0.5, 0.05, False),
    KrigingConfig("xy_ctx-patch_s4", "exponential", 16, 0.5, 0.01, True),
    KrigingConfig("xy_ctx-patch_s4", "exponential", 24, 0.5, 0.05, False),
    KrigingConfig("xy_ctx-patch_s4", "exponential", 32, 0.5, 0.05, False),
    KrigingConfig("xy_ctx-patch_s4", "exponential", 16, 0.75, 0.05, False),
    KrigingConfig("xy_ctx-patch_s4", "exponential", 24, 0.75, 0.05, False),
    KrigingConfig("xy_ctx-patch_s4", "exponential", 32, 0.75, 0.05, False),
    KrigingConfig("xy_ctx-patch_s4", "exponential", 16, 1.0, 0.05, False),
    KrigingConfig("xy_ctx-patch_s4", "exponential", 16, 1.5, 0.05, False),
    KrigingConfig("xy_ctx-patch_s4", "matern32", 24, 0.75, 0.001, True),
    KrigingConfig("xy_ctx-all_s4", "exponential", 32, 0.5, 0.1, False),
    KrigingConfig("xy_ctx-all_s4", "exponential", 32, 1.5, 0.05, False),
    KrigingConfig("xy_ctx-summary_s4", "exponential", 24, 0.75, 0.05, False),
)
ADVANCED_REPLACED_CONFIG = KrigingConfig(
    "xy_ctx-patch_s4", "exponential", 32, 0.5, 0.05, False
)


def active_group_configs(use_advanced_map: bool) -> tuple[KrigingConfig, ...]:
    if not use_advanced_map:
        return GROUP_CONFIGS
    return tuple(
        KrigingConfig(
            ADVANCED_MAP_METRIC,
            item.covariance,
            item.neighbors,
            item.bandwidth_scale,
            item.nugget,
            item.positive_weights,
        )
        if item == ADVANCED_REPLACED_CONFIG
        else item
        for item in GROUP_CONFIGS
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the optimized Physical-AI submission")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--features", type=Path, default=Path("artifacts/spectral_features.npy"))
    parser.add_argument(
        "--context", type=Path, default=Path("artifacts/map_context_advanced.npz")
    )
    parser.add_argument("--advanced-map-expert", action="store_true")
    parser.add_argument("--models", type=Path, nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--groupwise", type=Path, default=Path("artifacts/groupwise_kriging_v2.json"))
    parser.add_argument(
        "--conditional-groupwise",
        type=Path,
        default=Path("artifacts/conditional_groupwise_lowreg.json"),
    )
    parser.add_argument("--neural-blend", type=float, default=0.0001)
    parser.add_argument(
        "--disagreement-pas-model",
        type=Path,
        default=Path("artifacts/disagreement_gating_cv_pas_advanced_s075_e3.pt"),
    )
    parser.add_argument(
        "--disagreement-pdp-model",
        type=Path,
        default=Path("artifacts/disagreement_gating_cv_pdp_advanced_s06_e2.pt"),
    )
    parser.add_argument("--phase-neighbors", type=int, default=16)
    parser.add_argument("--phase-kriging-scale", type=float, default=0.75)
    parser.add_argument("--phase-kriging-nugget", type=float, default=0.1)
    parser.add_argument("--phase-sync-strength", type=float, default=0.2)
    parser.add_argument("--wavenumber", type=float, default=140.25)
    parser.add_argument("--phase-slope", type=float, default=0.0006)
    parser.add_argument("--h-steering-x", type=float, default=-1.75)
    parser.add_argument("--h-steering-y", type=float, default=-2.5)
    parser.add_argument("--v-steering-z", type=float, default=26.0)
    parser.add_argument("--energy-neighbors", type=int, default=4)
    parser.add_argument("--energy-gamma", type=float, default=0.2)
    parser.add_argument(
        "--group-energy", type=Path, default=Path("artifacts/channel_group_energy.npy")
    )
    parser.add_argument("--group-energy-neighbors", type=int, default=64)
    parser.add_argument("--group-energy-power", type=float, default=4.0)
    parser.add_argument("--group-energy-metric", type=str, default="xy_ctx-patch_s4")
    parser.add_argument(
        "--group-energy-advanced-metric", type=str, default=ADVANCED_MAP_METRIC
    )
    parser.add_argument("--group-energy-advanced-min-distance", type=float, default=1.2)
    parser.add_argument("--group-energy-advanced-max-distance", type=float, default=4.3)
    parser.add_argument("--group-energy-strength", type=float, default=0.3)
    parser.add_argument("--group-energy-mid-strength", type=float, default=0.5)
    parser.add_argument("--group-energy-mid-min-distance", type=float, default=1.6)
    parser.add_argument("--group-energy-mid-max-distance", type=float, default=3.5)
    parser.add_argument("--direct-blend", type=float, default=0.12)
    parser.add_argument("--scale-real", type=float, default=0.7945592076014179)
    parser.add_argument("--scale-imag", type=float, default=-0.004367502672896887)
    parser.add_argument(
        "--scalar-calibration",
        type=Path,
        default=Path("artifacts/v50_scalar_calibration.npz"),
    )
    parser.add_argument("--disable-scalar-calibration", action="store_true")
    parser.add_argument(
        "--local-scalar-ensemble",
        type=Path,
        default=Path("artifacts/v50_local_scalar_ensemble.npz"),
    )
    parser.add_argument("--disable-local-scalar-ensemble", action="store_true")
    parser.add_argument(
        "--ue-calibration-residual",
        type=Path,
        default=Path("artifacts/v50_ue_residual_calibration.npz"),
    )
    parser.add_argument("--disable-ue-calibration-residual", action="store_true")
    parser.add_argument(
        "--local-spectral-correction",
        type=Path,
        default=Path("artifacts/v50_geometry_ensemble_gate.npz"),
    )
    parser.add_argument("--disable-local-spectral-correction", action="store_true")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--relaxation", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def load_model(path: Path, device: torch.device) -> MapConditionedKernel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    model = MapConditionedKernel(
        state["position_mean"], state["position_std"], state["context_mean"], state["context_std"]
    )
    model.load_state_dict(state)
    return model.to(device).eval()


def load_expert_gate(path: Path, kind: str, device: torch.device) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    payload = checkpoint[kind]
    mean = torch.from_numpy(np.asarray(payload["condition_mean"], np.float32)).to(device)
    std = torch.from_numpy(np.asarray(payload["condition_std"], np.float32)).to(device)
    states = payload.get("state_dicts", [payload["state_dict"]] if "state_dict" in payload else [])
    models = []
    for state in states:
        model = ExpertDisagreementGate(
            mean.numel(), int(payload["groups"]), int(payload["experts"])
        ).to(device)
        model.load_state_dict(state)
        models.append(model.eval())
    return {
        "models": models,
        "mean": mean,
        "std": std,
        "scale": float(checkpoint["correction_scale"]),
        "path": str(path),
        "condition_mode": checkpoint.get("condition_mode", "basic"),
    }


@torch.inference_mode()
def neural_features(
    models: list[MapConditionedKernel],
    train_positions: np.ndarray,
    query_positions: np.ndarray,
    train_context: np.ndarray,
    query_context: np.ndarray,
    features: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    local, _ = nearest_neighbors(query_positions, train_positions, 16)
    neighbor = torch.from_numpy(local.astype(np.int64)).to(features.device)
    train_position_tensor = torch.from_numpy(train_positions.astype(np.float32)).to(features.device)
    query_position_tensor = torch.from_numpy(query_positions.astype(np.float32)).to(features.device)
    train_context_tensor = torch.from_numpy(train_context.astype(np.float32)).to(features.device)
    query_context_tensor = torch.from_numpy(query_context.astype(np.float32)).to(features.device)
    prediction = torch.zeros((len(query_positions), features.shape[1]), device=features.device)
    for model in models:
        for start in range(0, len(query_positions), batch_size):
            stop = min(start + batch_size, len(query_positions))
            source = neighbor[start:stop]
            pas_weight, pdp_weight = model(
                query_position_tensor[start:stop],
                query_context_tensor[start:stop],
                train_position_tensor[source],
                train_context_tensor[source],
            )
            prediction[start:stop] += interpolate_features(
                features[source], pas_weight, pdp_weight, 1024
            ) / len(models)
    return prediction.clamp_min(0.0)


@torch.inference_mode()
def predict_spectral_features(
    train_positions: np.ndarray,
    query_positions: np.ndarray,
    train_context: np.ndarray,
    query_context: np.ndarray,
    features: torch.Tensor,
    models: list[MapConditionedKernel],
    neural_blend: float,
    batch_size: int,
    group_weights: torch.Tensor,
    conditional_meta: dict,
    group_configs: tuple[KrigingConfig, ...] = GROUP_CONFIGS,
    pas_gate: dict[str, object] | None = None,
    pdp_gate: dict[str, object] | None = None,
) -> torch.Tensor:
    positions = np.concatenate((train_positions, query_positions), axis=0)
    contexts = np.concatenate((train_context, query_context), axis=0)
    embeddings = metric_embeddings(positions, contexts)
    train_indices = np.arange(len(train_positions), dtype=np.int64)
    query_indices = np.arange(len(train_positions), len(positions), dtype=np.int64)
    bank = torch.stack(
        [
            local_ordinary_kriging(
                config,
                embeddings[config.metric],
                train_indices,
                query_indices,
                features,
            )
            for config in group_configs
        ]
    )
    bank = graph_propagate(
        bank,
        embeddings["xy_y0.75"],
        train_indices,
        query_indices,
        features,
        neighbors=24,
        power=2.5,
        alpha=0.1,
    )
    pas_bank = bank[:, :, :1024].reshape(len(group_configs), -1, 256, 4)
    pdp_bank = bank[:, :, 1024:].reshape(len(group_configs), -1, 2, 4, 192)
    output_pas = torch.empty_like(pas_bank[0])
    output_pdp = torch.empty_like(pdp_bank[0])
    final_conditional = conditional_meta["final"]
    raw_condition = np.concatenate((query_positions[:, :2], query_context[:, :7]), axis=1)
    condition = torch.from_numpy(
        (
            raw_condition
            - np.asarray(final_conditional["condition_mean"], dtype=np.float32)
        )
        / np.asarray(final_conditional["condition_std"], dtype=np.float32)
    ).to(features.device)
    pas_corrections = torch.tensor(
        final_conditional["pas_corrections"], dtype=torch.float32, device=features.device
    )
    pas_logits = torch.stack(
        [
            torch.log(group_weights[ue].clamp_min(1e-7))[None]
            + condition @ pas_corrections[ue]
            for ue in range(4)
        ],
        dim=1,
    )
    pas_expert = pas_bank.permute(1, 3, 0, 2)
    if pas_gate is not None:
        raw_gate_condition = expert_gate_condition(
            query_positions, query_context, pas_gate["condition_mode"]
        )
        gate_condition = (
            torch.from_numpy(raw_gate_condition).to(features.device) - pas_gate["mean"]
        ) / pas_gate["std"]
        correction = torch.stack(
            [model(gate_condition, pas_expert) for model in pas_gate["models"]]
        ).mean(0)
        pas_logits = pas_logits + pas_gate["scale"] * correction
    output_pas = torch.einsum(
        "quc,qucm->qmu", torch.softmax(pas_logits, dim=2), pas_expert
    )
    pdp_expert = pdp_bank.permute(1, 2, 3, 0, 4).reshape(
        len(query_positions), 8, len(group_configs), 192
    )
    pdp_logits = torch.log(group_weights[4:].clamp_min(1e-7))[None].expand(
        len(query_positions), -1, -1
    )
    if pdp_gate is not None:
        raw_gate_condition = expert_gate_condition(
            query_positions, query_context, pdp_gate["condition_mode"]
        )
        gate_condition = (
            torch.from_numpy(raw_gate_condition).to(features.device) - pdp_gate["mean"]
        ) / pdp_gate["std"]
        correction = torch.stack(
            [model(gate_condition, pdp_expert) for model in pdp_gate["models"]]
        ).mean(0)
        pdp_logits = pdp_logits + pdp_gate["scale"] * correction
    output_pdp = torch.einsum(
        "qgc,qgcs->qgs", torch.softmax(pdp_logits, dim=2), pdp_expert
    ).reshape(len(query_positions), 2, 4, 192)
    compact = torch.cat((output_pas.flatten(1), output_pdp.flatten(1)), dim=1)
    if neural_blend > 0.0:
        neural_context_dim = int(models[0].context_mean.numel())
        learned = neural_features(
            models,
            train_positions,
            query_positions,
            train_context[:, :neural_context_dim],
            query_context[:, :neural_context_dim],
            features,
            batch_size,
        )
        compact = compact.lerp(learned, neural_blend)
    return compact


@torch.inference_mode()
def reconstruct_batch(
    direct: torch.Tensor,
    compact: torch.Tensor,
    dims,
    iterations: int,
    relaxation: float,
    direct_blend: float,
    complex_scale: complex,
    neighbor_energy: torch.Tensor,
    energy_neighbors: int,
    energy_gamma: float,
    desired_group_fraction: torch.Tensor,
    group_energy_strength: float | torch.Tensor,
    return_group_energy: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    target_pas, target_pdp = spectral_targets_from_features(compact, dims)
    channel = direct
    for _ in range(iterations):
        channel = replace_fourier_magnitude(channel, target_pas, dim=1, relaxation=relaxation)
        channel = replace_fourier_magnitude(channel, target_pdp, dim=-1, relaxation=relaxation)
    direct_energy = torch.sum(torch.abs(direct).square(), dim=(1, 2, 3), keepdim=True)
    projected_energy = torch.sum(torch.abs(channel).square(), dim=(1, 2, 3), keepdim=True)
    energy_scale = torch.sqrt(direct_energy / projected_energy.clamp_min(1e-30))
    blended = (1.0 - direct_blend) * energy_scale * channel + direct_blend * direct
    geometric_energy = torch.exp(
        torch.mean(
            torch.log(neighbor_energy[:, :energy_neighbors].clamp_min(1e-30)), dim=1
        )
    ).to(direct.real.dtype)
    correction = (
        geometric_energy[:, None, None, None] / direct_energy.clamp_min(1e-30)
    ).pow(energy_gamma / 2.0)
    enabled = (
        bool(torch.any(group_energy_strength > 0.0))
        if isinstance(group_energy_strength, torch.Tensor)
        else group_energy_strength > 0.0
    )
    grouped = blended.reshape(
        len(blended),
        dims.bs_polarizations,
        dims.bs_h * dims.bs_v,
        dims.ue_antennas,
        dims.subcarriers,
    )
    current_group_energy = torch.sum(torch.abs(grouped).square(), dim=(2, 4))
    if enabled:
        current_fraction = current_group_energy / current_group_energy.sum(
            dim=(1, 2), keepdim=True
        ).clamp_min(1e-30)
        exponent = group_energy_strength / 2.0
        if isinstance(exponent, torch.Tensor):
            exponent = exponent[:, None, None]
        group_scale = (
            desired_group_fraction / current_fraction.clamp_min(1e-12)
        ).pow(exponent)
        blended = (grouped * group_scale[:, :, None, :, None]).reshape_as(blended)
    prediction = blended * correction * complex_scale
    if return_group_energy:
        return prediction, current_group_energy
    return prediction


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.neural_blend <= 1.0:
        raise ValueError("--neural-blend must be in [0, 1]")
    if not 0.0 <= args.direct_blend <= 1.0:
        raise ValueError("--direct-blend must be in [0, 1]")
    device = resolve_device(args.device)
    data = RoundData(args.root)
    data.validate()
    output_path = args.output or data.output_path
    all_train_positions = np.asarray(data.train_positions, dtype=np.float32)
    test_positions = np.asarray(data.test_positions, dtype=np.float32)
    context_archive = np.load(args.context)
    all_train_context = context_archive["train"].astype(np.float32)
    test_context = context_archive["test"].astype(np.float32)
    all_features = np.asarray(np.load(args.features, mmap_mode="r"), dtype=np.float32)
    valid = nonzero_feature_indices(all_features)
    excluded = np.setdiff1d(np.arange(len(all_features)), valid)
    train_positions = all_train_positions[valid]
    train_context = all_train_context[valid]
    group_configs = active_group_configs(args.advanced_map_expert)
    if args.advanced_map_expert and ADVANCED_MAP_METRIC not in metric_embeddings(
        np.concatenate((train_positions, test_positions), axis=0),
        np.concatenate((train_context, test_context), axis=0),
    ):
        raise ValueError(
            f"--advanced-map-expert requires the 362-column advanced context; got {train_context.shape[1]}"
        )
    feature_bank = torch.from_numpy(all_features[valid].copy()).to(device)
    models = [load_model(path, device) for path in args.models]
    pas_gate = load_expert_gate(args.disagreement_pas_model, "pas", device)
    pdp_gate = load_expert_gate(args.disagreement_pdp_model, "pdp", device)
    groupwise_meta = json.loads(args.groupwise.read_text(encoding="utf-8"))
    conditional_meta = json.loads(args.conditional_groupwise.read_text(encoding="utf-8"))
    group_weights = torch.tensor(groupwise_meta["weights"], device=device)
    compact = predict_spectral_features(
        train_positions,
        test_positions,
        train_context,
        test_context,
        feature_bank,
        models,
        args.neural_blend,
        args.batch_size,
        group_weights,
        conditional_meta,
        group_configs,
        pas_gate,
        pdp_gate,
    )
    local_spectral_correction = (
        None
        if args.disable_local_spectral_correction
        else LocalSpectralCorrection.load(args.local_spectral_correction)
    )
    if local_spectral_correction is not None:
        compact = local_spectral_correction.apply(
            compact, test_positions, test_context
        )

    all_group_energy = np.load(args.group_energy)
    group_fraction = all_group_energy[valid] / np.maximum(
        all_group_energy[valid].sum(axis=(1, 2), keepdims=True), 1e-30
    )
    combined_positions = np.concatenate((train_positions, test_positions), axis=0)
    combined_context = np.concatenate((train_context, test_context), axis=0)
    phase_local, phase_distance = nearest_neighbors(
        test_positions, train_positions, args.phase_neighbors
    )

    energy_embeddings = metric_embeddings(combined_positions, combined_context)

    def interpolate_group_fraction(metric: str) -> np.ndarray:
        if metric not in energy_embeddings:
            raise ValueError(
                f"Group-energy metric {metric!r} is unavailable for "
                f"{combined_context.shape[1]} context columns"
            )
        embedding = energy_embeddings[metric]
        local, distance = nearest_neighbors(
            embedding[len(train_positions) :],
            embedding[: len(train_positions)],
            args.group_energy_neighbors,
        )
        weight = (distance + 1e-3) ** (-args.group_energy_power)
        weight /= weight.sum(axis=1, keepdims=True)
        prediction = np.exp(
            np.sum(
                weight[:, :, None, None]
                * np.log(group_fraction[local].clip(1e-12)),
                axis=1,
            )
        )
        return prediction / prediction.sum(axis=(1, 2), keepdims=True)

    legacy_group_fraction = interpolate_group_fraction(args.group_energy_metric)
    advanced_group_fraction = interpolate_group_fraction(
        args.group_energy_advanced_metric
    )
    use_advanced_group_energy = (
        (phase_distance[:, 0] >= args.group_energy_advanced_min_distance)
        & (phase_distance[:, 0] < args.group_energy_advanced_max_distance)
    )
    predicted_group_fraction = np.where(
        use_advanced_group_energy[:, None, None],
        advanced_group_fraction,
        legacy_group_fraction,
    )
    adaptive_group_energy_strength = np.where(
        (phase_distance[:, 0] >= args.group_energy_mid_min_distance)
        & (phase_distance[:, 0] < args.group_energy_mid_max_distance),
        args.group_energy_mid_strength,
        args.group_energy_strength,
    ).astype(np.float32)
    weak_phase_weight = ordinary_kriging_weights(
        test_positions[:, :2],
        train_positions[phase_local, :2],
        phase_distance,
        2.0,
        0.001,
        positive=False,
    )
    strong_phase_weight = ordinary_kriging_weights(
        test_positions[:, :2],
        train_positions[phase_local, :2],
        phase_distance,
        2.0,
        0.1,
        positive=False,
    )
    strong_group = np.array(
        [[False, False, True, True], [True, True, False, False]]
    )
    phase_weight = np.where(
        strong_group[None, None],
        strong_phase_weight[:, :, None, None],
        weak_phase_weight[:, :, None, None],
    )
    bs_position = np.asarray(data.dims.bs_position, dtype=np.float64)
    train_radius = np.linalg.norm(train_positions - bs_position, axis=1)
    test_radius = np.linalg.norm(test_positions - bs_position, axis=1)
    radial_delta = train_radius[phase_local] - test_radius[:, None]
    train_direction = (train_positions - bs_position) / train_radius[:, None]
    test_direction = (test_positions - bs_position) / test_radius[:, None]
    direction_delta = train_direction[phase_local] - test_direction[:, None, :]
    subcarrier_offset = (
        torch.arange(data.dims.subcarriers, device=device, dtype=torch.float32)
        - (data.dims.subcarriers - 1) / 2.0
    )
    h_index = torch.arange(data.dims.bs_h, device=device, dtype=torch.float32)
    h_index -= (data.dims.bs_h - 1) / 2.0
    v_index = torch.arange(data.dims.bs_v, device=device, dtype=torch.float32)
    v_index -= (data.dims.bs_v - 1) / 2.0
    h_grid = h_index[:, None].expand(data.dims.bs_h, data.dims.bs_v).reshape(-1)
    v_grid = v_index[None, :].expand(data.dims.bs_h, data.dims.bs_v).reshape(-1)
    h_grid = h_grid.repeat(data.dims.bs_polarizations)
    v_grid = v_grid.repeat(data.dims.bs_polarizations)
    scalar_calibration = (
        None
        if args.disable_scalar_calibration
        else ScalarCalibration.load(args.scalar_calibration)
    )
    local_scalar_ensemble = (
        None
        if scalar_calibration is None or args.disable_local_scalar_ensemble
        else LocalScalarEnsemble.load(args.local_scalar_ensemble)
    )
    ue_calibration_residual = (
        None
        if scalar_calibration is None or args.disable_ue_calibration_residual
        else UECalibrationResidual.load(args.ue_calibration_residual)
    )
    complex_scale = (
        1.0 + 0.0j
        if scalar_calibration is not None
        else complex(args.scale_real, args.scale_imag)
    )
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.complex64,
        shape=(len(test_positions), *data.dims.channel_shape),
    )
    for start in range(0, len(test_positions), args.batch_size):
        stop = min(start + args.batch_size, len(test_positions))
        source_global = valid[phase_local[start:stop]]
        source = torch.from_numpy(
            np.array(data.train_channels[source_global], dtype=np.complex64, copy=True)
        ).to(device)
        delta = torch.from_numpy(radial_delta[start:stop].astype(np.float32)).to(device)
        angular_delta = torch.from_numpy(
            direction_delta[start:stop].astype(np.float32)
        ).to(device)
        weight = torch.from_numpy(phase_weight[start:stop].astype(np.float32)).to(device)
        wavenumber = args.wavenumber + args.phase_slope * subcarrier_offset
        radial_phase = torch.exp(1j * delta[:, :, None] * wavenumber)
        steering = torch.exp(
            1j
            * (
                (
                    args.h_steering_x * angular_delta[:, :, 0, None]
                    + args.h_steering_y * angular_delta[:, :, 1, None]
                )
                * h_grid
                + args.v_steering_z * angular_delta[:, :, 2, None] * v_grid
            )
        )
        source_group = source.reshape(
            stop - start,
            args.phase_neighbors,
            data.dims.bs_polarizations,
            data.dims.bs_h * data.dims.bs_v,
            data.dims.ue_antennas,
            data.dims.subcarriers,
        )
        steering_group = steering.reshape(
            stop - start,
            args.phase_neighbors,
            data.dims.bs_polarizations,
            data.dims.bs_h * data.dims.bs_v,
        )
        adjusted_group = (
            source_group
            * steering_group[:, :, :, :, None, None]
            * radial_phase[:, :, None, None, None, :]
        )
        if args.phase_sync_strength != 0.0:
            gram = torch.einsum(
                "bkpaus,blpaus->bpukl", torch.conj(adjusted_group), adjusted_group
            )
            diagonal = torch.diagonal(gram, dim1=-2, dim2=-1).real.clamp_min(1e-30)
            coherence = gram / torch.sqrt(
                diagonal[..., :, None] * diagonal[..., None, :]
            )
            vectors = torch.linalg.eigh(coherence).eigenvectors[..., -1]
            vectors *= torch.exp(-1j * torch.angle(vectors[..., :1]))
            sync = vectors.permute(0, 3, 1, 2)
            weight = weight * torch.exp(
                1j * args.phase_sync_strength * torch.angle(sync)
            )
        direct = torch.sum(
            adjusted_group * weight[:, :, :, None, :, None], dim=1
        ).reshape(stop - start, *data.dims.channel_shape)
        neighbor_energy = torch.sum(
            torch.abs(source).square(), dim=(2, 3, 4), dtype=torch.float64
        )
        reconstruction = reconstruct_batch(
            direct,
            compact[start:stop],
            data.dims,
            args.iterations,
            args.relaxation,
            args.direct_blend,
            complex_scale,
            neighbor_energy,
            args.energy_neighbors,
            args.energy_gamma,
            torch.from_numpy(
                predicted_group_fraction[start:stop].astype(np.float32)
            ).to(device),
            torch.from_numpy(adaptive_group_energy_strength[start:stop]).to(device),
            return_group_energy=scalar_calibration is not None,
        )
        if scalar_calibration is not None:
            prediction, pre_group_energy = reconstruction
            final_pred_energy = torch.sum(
                torch.abs(prediction).square(),
                dim=(1, 2, 3),
                dtype=torch.float64,
            ).cpu().numpy()
            sample_scale = scalar_calibration.predict(
                test_positions[start:stop],
                test_context[start:stop],
                phase_distance[start:stop, 0],
                final_pred_energy,
                pre_group_energy.cpu().numpy(),
            )
            if local_scalar_ensemble is not None:
                sample_scale = sample_scale + local_scalar_ensemble.predict(
                    test_positions[start:stop], test_context[start:stop]
                )
            if ue_calibration_residual is None:
                scale_tensor = torch.from_numpy(
                    sample_scale.astype(np.complex64, copy=False)
                ).to(device)
                prediction = prediction * scale_tensor[:, None, None, None]
            else:
                ue_scale = sample_scale[:, None] + ue_calibration_residual.predict(
                    test_positions[start:stop],
                    test_context[start:stop],
                    phase_distance[start:stop, 0],
                    final_pred_energy,
                    pre_group_energy.cpu().numpy(),
                )
                scale_tensor = torch.from_numpy(
                    ue_scale.astype(np.complex64, copy=False)
                ).to(device)
                prediction = prediction * scale_tensor[:, None, :, None]
        else:
            prediction = reconstruction
        output[start:stop] = prediction.cpu().numpy().astype(np.complex64, copy=False)
        output.flush()
        print(f"generated {stop}/{len(test_positions)} on {device}", flush=True)

    metadata = {
        "version": "v50-geometry-ensemble-four-action-sample-gating",
        "output": str(output_path),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "device": str(device),
        "valid_training_samples": int(len(valid)),
        "excluded_zero_channel_indices": excluded.tolist(),
        "neural_models": [str(path) for path in args.models],
        "neural_blend": args.neural_blend,
        "conditional_groupwise": str(args.conditional_groupwise),
        "context": str(args.context),
        "advanced_map_expert": args.advanced_map_expert,
        "expert_metrics": [config.metric for config in group_configs],
        "disagreement_pas_model": str(args.disagreement_pas_model),
        "disagreement_pdp_model": str(args.disagreement_pdp_model),
        "phase_neighbors": args.phase_neighbors,
        "phase_weighting": "groupwise ordinary kriging with local phase synchronization",
        "phase_kriging_scale": args.phase_kriging_scale,
        "phase_kriging_nugget": args.phase_kriging_nugget,
        "phase_groupwise_kernels": {
            "weak": {"neighbors": 16, "scale": 2.0, "nugget": 0.001},
            "strong": {"neighbors": 16, "scale": 2.0, "nugget": 0.1},
        },
        "phase_sync_strength": args.phase_sync_strength,
        "wavenumber": args.wavenumber,
        "phase_slope": args.phase_slope,
        "bs_array_steering": [args.h_steering_x, args.h_steering_y, args.v_steering_z],
        "direct_blend": args.direct_blend,
        "energy_neighbors": args.energy_neighbors,
        "energy_gamma": args.energy_gamma,
        "group_energy": str(args.group_energy),
        "group_energy_neighbors": args.group_energy_neighbors,
        "group_energy_power": args.group_energy_power,
        "group_energy_metric": args.group_energy_metric,
        "group_energy_advanced_metric": args.group_energy_advanced_metric,
        "group_energy_advanced_distance": [
            args.group_energy_advanced_min_distance,
            args.group_energy_advanced_max_distance,
        ],
        "group_energy_strength": args.group_energy_strength,
        "group_energy_mid_strength": args.group_energy_mid_strength,
        "group_energy_mid_distance": [
            args.group_energy_mid_min_distance,
            args.group_energy_mid_max_distance,
        ],
        "complex_scale": [args.scale_real, args.scale_imag],
        "scalar_calibration": (
            None if scalar_calibration is None else str(args.scalar_calibration)
        ),
        "scalar_calibration_mode": (
            None if scalar_calibration is None else scalar_calibration.mode
        ),
        "scalar_calibration_strength": (
            None if scalar_calibration is None else scalar_calibration.strength
        ),
        "local_scalar_ensemble": (
            None
            if local_scalar_ensemble is None
            else str(args.local_scalar_ensemble)
        ),
        "local_scalar_metrics": (
            None
            if local_scalar_ensemble is None
            else list(local_scalar_ensemble.metrics)
        ),
        "local_scalar_blend_weight": (
            None
            if local_scalar_ensemble is None
            else local_scalar_ensemble.blend_weight.tolist()
        ),
        "local_scalar_clip": (
            None if local_scalar_ensemble is None else local_scalar_ensemble.clip
        ),
        "local_spectral_correction": (
            None
            if local_spectral_correction is None
            else str(args.local_spectral_correction)
        ),
        "local_spectral_pas": (
            None
            if local_spectral_correction is None
            else {
                "metric": local_spectral_correction.pas.metric,
                "neighbors": local_spectral_correction.pas.neighbors,
                "method": local_spectral_correction.pas.method,
                "strength": local_spectral_correction.pas.strength,
                "gate": local_spectral_correction.pas.gate_name,
                "gate_direction": local_spectral_correction.pas.gate_direction,
                "gate_threshold": local_spectral_correction.pas.gate_threshold,
            }
        ),
        "local_spectral_pdp": (
            None
            if local_spectral_correction is None
            else {
                "metric": local_spectral_correction.pdp.metric,
                "neighbors": local_spectral_correction.pdp.neighbors,
                "method": local_spectral_correction.pdp.method,
                "strength": local_spectral_correction.pdp.strength,
                "gate": local_spectral_correction.pdp.gate_name,
                "gate_direction": local_spectral_correction.pdp.gate_direction,
                "gate_threshold": local_spectral_correction.pdp.gate_threshold,
            }
        ),
        "joint_sample_spectral_gate": (
            None
            if local_spectral_correction is None
            or local_spectral_correction.sample_gate_coefficient is None
            else {
                "objective": "joint PAS/PDP/optimal-NMSE gain",
                "selection_fraction": local_spectral_correction.sample_gate_fraction,
                "feature_width": int(
                    local_spectral_correction.sample_gate_coefficient.size - 1
                ),
            }
        ),
        "four_action_sample_spectral_gate": (
            None
            if local_spectral_correction is None
            or local_spectral_correction.sample_action_physical_reference is None
            else {
                "objective": "joint PAS/PDP/optimal-NMSE gain",
                "actions": ["none", "PAS", "PDP", "PAS+PDP"],
                "selection_fraction": (
                    local_spectral_correction.sample_action_physical_fraction
                ),
                "primary_metric": (
                    local_spectral_correction.sample_action_physical_metric
                ),
                "primary_neighbors": (
                    local_spectral_correction.sample_action_physical_neighbors
                ),
                "secondary_metric": (
                    local_spectral_correction.sample_action_physical_secondary_metric
                ),
                "secondary_neighbors": (
                    local_spectral_correction.sample_action_physical_secondary_neighbors
                ),
                "primary_ensemble_weight": (
                    local_spectral_correction.sample_action_physical_ensemble_weight
                ),
                "action_bias": (
                    None
                    if local_spectral_correction.sample_action_physical_bias is None
                    else local_spectral_correction.sample_action_physical_bias.tolist()
                ),
            }
        ),
        "ue_calibration_residual": (
            None
            if ue_calibration_residual is None
            else str(args.ue_calibration_residual)
        ),
        "ue_calibration_residual_strength": (
            None
            if ue_calibration_residual is None
            else ue_calibration_residual.strength
        ),
        "iterations": args.iterations,
        "relaxation": args.relaxation,
        "cross_validated_metrics": {
            "pas": 0.7648172497749328,
            "pdp": 0.8519760847091675,
            "nmse": 0.614336041706267,
            "score": 0.7706072766760469,
        },
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
