from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from physical_ai.data import RoundData
from physical_ai.delay_attention import (
    DelaywiseNeighborAttention,
    angle_delay_coefficients,
    observable_delay_statistics,
    phase_aligned_idw,
    reconstruct_from_attention,
)
from physical_ai.features import nonzero_feature_indices, spectral_targets_from_features
from physical_ai.expert_gate import ExpertDisagreementGate, expert_gate_condition
from physical_ai.metrics import cosine_similarity_last, pas_spectrum, pdp_spectrum
from physical_ai.model import MapConditionedKernel, interpolate_features
from physical_ai.neighbors import distance_weights, nearest_neighbors
from physical_ai.spectral_calibration import LocalSpectralCorrection
from physical_ai.spatial import ADVANCED_MAP_METRIC, metric_embeddings
from experiments.search_improved_graph_gpu import graph_matrices
from experiments.search_kriging_ensemble_gpu import CONFIGS, Config, predict_config
from experiments.search_groupwise_kriging_ensemble import mix_groups
from experiments.search_v4_reconstruction_gpu import replace_magnitude


PAS_CONFIG = Config("xy_ctx-patch_s4", "exponential", 16, 0.5, 0.01, True)
PDP_CONFIG = Config("xy_ctx-patch_s4", "exponential", 24, 0.75, 0.05, False)
ADVANCED_REPLACED_CONFIG = Config(
    "xy_ctx-patch_s4", "exponential", 32, 0.5, 0.05, False
)


def active_group_configs(use_advanced_map: bool) -> tuple[Config, ...]:
    if not use_advanced_map:
        return CONFIGS
    return tuple(
        Config(ADVANCED_MAP_METRIC, item.kind, item.k, item.scale, item.nugget, item.positive)
        if item == ADVANCED_REPLACED_CONFIG
        else item
        for item in CONFIGS
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V5 radial carrier phase validation")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/cv_noout_split20260804.pt"))
    parser.add_argument("--context", type=Path, default=Path("artifacts/map_context.npz"))
    parser.add_argument("--advanced-map-expert", action="store_true")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--relaxation", type=float, default=0.75)
    parser.add_argument("--spectral-refine-steps", type=int, default=0)
    parser.add_argument("--spectral-refine-lr", type=float, default=0.02)
    parser.add_argument("--spectral-refine-anchor", type=float, default=0.0)
    parser.add_argument("--wavenumbers", type=float, nargs="+", default=[0.0, 139.75, 140.0, 140.25, 140.5, 140.75])
    parser.add_argument("--blend-wavenumber", type=float, default=140.25)
    parser.add_argument("--blend-scale-real", type=float, default=9.896153682415038e-5)
    parser.add_argument("--blend-scale-imag", type=float, default=-3.0655571541937293e-6)
    parser.add_argument("--phase-neighbors", type=int, default=16)
    parser.add_argument("--phase-power", type=float, default=2.0)
    parser.add_argument("--phase-softening", type=float, default=0.0)
    parser.add_argument("--phase-weighting", choices=("idw", "kriging"), default="idw")
    parser.add_argument("--phase-kriging-scale", type=float, default=0.75)
    parser.add_argument("--phase-kriging-nugget", type=float, default=0.1)
    parser.add_argument("--phase-neighbor-metric", type=str, default="xy")
    parser.add_argument("--phase-groupwise", action="store_true")
    parser.add_argument("--phase-weak8", action="store_true")
    parser.add_argument("--phase-sync-strength", type=float, default=0.0)
    parser.add_argument("--delay-sync-strength", type=float, default=0.0)
    parser.add_argument("--delaywise-direct-strength", type=float, default=0.0)
    parser.add_argument("--delaywise-neighbors", type=int, default=8)
    parser.add_argument("--delaywise-power", type=float, default=2.0)
    parser.add_argument("--delaywise-softening", type=float, default=0.0)
    parser.add_argument("--delaywise-coherence", type=float, default=1.0)
    parser.add_argument("--delaywise-energy", type=float, default=0.0)
    parser.add_argument("--delaywise-fusion", choices=("magnitude", "power", "complex"), default="magnitude")
    parser.add_argument("--delaywise-angle-transform", choices=("flat", "hv"), default="flat")
    parser.add_argument("--delaywise-alignment", choices=("none", "norm", "ls"), default="norm")
    parser.add_argument("--delaywise-hidw-blend", type=float, default=0.1)
    parser.add_argument("--learned-delaywise-model", type=Path, default=None)
    parser.add_argument("--learned-delaywise-strength", type=float, default=0.0)
    parser.add_argument("--evaluate-hidw-alignment", action="store_true")
    parser.add_argument("--phase-slope", type=float, default=0.0)
    parser.add_argument("--neural-blend", type=float, default=0.0)
    parser.add_argument("--disagreement-pas-model", type=Path, default=None)
    parser.add_argument("--disagreement-pdp-model", type=Path, default=None)
    parser.add_argument("--h-steering", type=float, default=0.0)
    parser.add_argument("--h-steering-x", type=float, default=0.0)
    parser.add_argument("--v-steering", type=float, default=0.0)
    parser.add_argument("--energy-neighbors", type=int, default=4)
    parser.add_argument("--energy-gamma", type=float, default=0.0)
    parser.add_argument("--terminal-pdp", type=float, default=0.0)
    parser.add_argument("--groupwise", type=Path, default=None)
    parser.add_argument("--conditional-groupwise", type=Path, default=None)
    parser.add_argument("--post-projection", type=float, default=0.0)
    parser.add_argument("--pas-pca-rank", type=int, default=0)
    parser.add_argument("--pas-pca-blend", type=float, default=0.0)
    parser.add_argument("--sample-stats", type=Path, default=None)
    parser.add_argument("--local-spectral-correction", type=Path, default=None)
    parser.add_argument("--sample-stats-group-energy-strength", type=float, default=None)
    parser.add_argument("--sample-stats-group-energy-mid-strength", type=float, default=0.5)
    parser.add_argument("--sample-stats-group-energy-mid-min-distance", type=float, default=1.6)
    parser.add_argument("--sample-stats-group-energy-mid-max-distance", type=float, default=3.5)
    parser.add_argument("--diagnostic-beta", type=float, default=0.14)
    parser.add_argument("--pol-ue-calibration", type=Path, default=None)
    parser.add_argument(
        "--pol-ue-strengths",
        type=float,
        nargs="+",
        default=(0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--group-energy", type=Path, default=None)
    parser.add_argument("--group-energy-metric", type=str, default="xy_ctx-patch_s4")
    parser.add_argument("--group-energy-advanced-metric", type=str, default=None)
    parser.add_argument("--group-energy-advanced-min-distance", type=float, default=1.2)
    parser.add_argument("--group-energy-advanced-max-distance", type=float, default=4.3)
    parser.add_argument("--group-energy-neighbors", type=int, default=64)
    parser.add_argument("--group-energy-power", type=float, default=4.0)
    parser.add_argument(
        "--adaptive-edges",
        type=float,
        nargs="+",
        default=(0.0, 1.5, 2.3, 3.3, float("inf")),
    )
    parser.add_argument(
        "--group-energy-strengths",
        type=float,
        nargs="+",
        default=(0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--power-marginals", type=Path, default=None)
    parser.add_argument(
        "--marginal-strengths",
        type=float,
        nargs="+",
        default=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/v5_phase_eval.json"))
    return parser.parse_args()


def refine_spectral_cosine(
    channel: torch.Tensor,
    target_pas: torch.Tensor,
    target_pdp: torch.Tensor,
    dims,
    steps: int,
    learning_rate: float,
    anchor_strength: float,
) -> torch.Tensor:
    """Directly optimize the two spectral cosine objectives after projection."""
    if steps <= 0:
        return channel
    with torch.enable_grad():
        rms = torch.sqrt(
            torch.mean(torch.abs(channel).square(), dim=(1, 2, 3), keepdim=True)
        ).clamp_min(1e-20)
        initial = (channel / rms).detach()
        variable = initial.clone().requires_grad_(True)
        optimizer = torch.optim.Adam((variable,), lr=learning_rate)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            predicted_pas = pas_spectrum(variable, dims)
            predicted_pdp = pdp_spectrum(variable)
            pas_similarity = cosine_similarity_last(
                predicted_pas, target_pas.permute(0, 2, 3, 1)
            ).mean()
            pdp_similarity = cosine_similarity_last(predicted_pdp, target_pdp).mean()
            anchor = torch.mean(torch.abs(variable - initial).square())
            loss = -0.5 * (pas_similarity + pdp_similarity) + anchor_strength * anchor
            loss.backward()
            optimizer.step()
        return (variable.detach() * rms).to(channel.dtype)


def delaywise_attention_prediction(
    adjusted_group: torch.Tensor,
    distances: torch.Tensor,
    neighbors: int,
    power: float,
    softening: float,
    coherence_gamma: float,
    energy_gamma: float,
    fusion: str,
    alignment: str,
    hidw_blend: float,
    angle_transform: str,
    bs_h: int,
    bs_v: int,
) -> torch.Tensor:
    """Fuse steered neighbors delay-wise while retaining nearest-neighbor phase."""
    source = adjusted_group[:, :neighbors].flatten(2, 3)
    spatial = (distances[:, :neighbors] + softening).clamp_min(1e-6).pow(-power)
    spatial /= spatial.sum(1, keepdim=True).clamp_min(1e-20)
    anchor = source[:, :1]
    cross = torch.sum(torch.conj(source) * anchor, dim=(2, 3, 4), keepdim=True)
    phase_align = cross / torch.abs(cross).clamp_min(1e-30)
    aligned_source = source * phase_align
    hidw = torch.sum(aligned_source * spatial[:, :, None, None, None], dim=1)
    if angle_transform == "flat":
        coefficient = torch.fft.fft(
            torch.fft.fft(aligned_source, dim=2, norm="ortho"),
            dim=-1,
            norm="ortho",
        )
        reduction_dims = (2, 3)
    else:
        batch, count, antennas, ue, subcarriers = aligned_source.shape
        polarizations = antennas // (bs_h * bs_v)
        layout = aligned_source.reshape(
            batch, count, polarizations, bs_h, bs_v, ue, subcarriers
        )
        coefficient = torch.fft.fft(
            torch.fft.fft(
                torch.fft.fft(layout, dim=3, norm="ortho"),
                dim=4,
                norm="ortho",
            ),
            dim=-1,
            norm="ortho",
        )
        reduction_dims = (2, 3, 4, 5)
    coefficient_energy = torch.sum(
        torch.abs(coefficient).square(), dim=reduction_dims
    ).clamp_min(1e-30)
    anchor_coefficient = coefficient[:, :1]
    coherence = torch.abs(
        torch.sum(
            torch.conj(coefficient) * anchor_coefficient, dim=reduction_dims
        )
    ) / torch.sqrt(coefficient_energy * coefficient_energy[:, :1]).clamp_min(1e-30)
    coherence_z = (coherence - coherence.mean(1, keepdim=True)) / coherence.std(
        1, keepdim=True
    ).clamp_min(1e-4)
    log_energy = torch.log(coefficient_energy)
    energy_z = (log_energy - log_energy.mean(1, keepdim=True)) / log_energy.std(
        1, keepdim=True
    ).clamp_min(1e-4)
    delay_weight = torch.softmax(
        torch.log(spatial.clamp_min(1e-20))[:, :, None]
        + coherence_gamma * coherence_z
        + energy_gamma * energy_z,
        dim=1,
    )
    expanded = delay_weight[
        (...,) + (None,) * (coefficient.ndim - 3) + (slice(None),)
    ]
    if fusion == "magnitude":
        amplitude = torch.sum(expanded * torch.abs(coefficient), dim=1)
    elif fusion == "power":
        amplitude = torch.sqrt(
            torch.sum(expanded * torch.abs(coefficient).square(), dim=1).clamp_min(0.0)
        )
    else:
        amplitude = torch.abs(torch.sum(expanded * coefficient, dim=1))
    nearest_phase = anchor_coefficient[:, 0] / torch.abs(anchor_coefficient[:, 0]).clamp_min(1e-30)
    if angle_transform == "flat":
        fused = torch.fft.ifft(
            torch.fft.ifft(amplitude * nearest_phase, dim=-1, norm="ortho"),
            dim=1,
            norm="ortho",
        )
    else:
        fused = torch.fft.ifft(
            torch.fft.ifft(
                torch.fft.ifft(
                    amplitude * nearest_phase, dim=-1, norm="ortho"
                ),
                dim=3,
                norm="ortho",
            ),
            dim=2,
            norm="ortho",
        ).reshape(batch, antennas, ue, subcarriers)
    axes = (1, 2, 3)
    alignment_cross = torch.sum(torch.conj(fused) * hidw, dim=axes, keepdim=True)
    fused_energy = torch.sum(torch.abs(fused).square(), dim=axes, keepdim=True).clamp_min(1e-30)
    if alignment == "ls":
        fused = fused * alignment_cross / fused_energy
    elif alignment == "norm":
        hidw_energy = torch.sum(torch.abs(hidw).square(), dim=axes, keepdim=True)
        fused = fused * torch.sqrt(hidw_energy / fused_energy) * (
            alignment_cross / torch.abs(alignment_cross).clamp_min(1e-30)
        )
    return (1.0 - hidw_blend) * fused + hidw_blend * hidw


def make_delaywise_pair_features(
    positions: np.ndarray,
    contexts: np.ndarray,
    query: np.ndarray,
    neighbors: np.ndarray,
    distances: np.ndarray,
    checkpoint: dict,
) -> np.ndarray:
    """Reproduce the geometric/context pair features used during attention training."""
    position_mean = np.asarray(checkpoint["position_mean"], dtype=np.float32)
    position_std = np.asarray(checkpoint["position_std"], dtype=np.float32)
    context_mean = np.asarray(checkpoint["context_mean"], dtype=np.float32)
    context_std = np.asarray(checkpoint["context_std"], dtype=np.float32)
    qpos = (positions[query, :2] - position_mean) / position_std
    delta = (positions[neighbors, :2] - positions[query, None, :2]) / position_std
    unit = positions[neighbors, :2] - positions[query, None, :2]
    unit /= np.maximum(distances[..., None], 1e-6)
    qctx = (contexts[query, :7] - context_mean) / context_std
    cdelta = (contexts[neighbors, :7] - contexts[query, None, :7]) / context_std
    rank = np.arange(neighbors.shape[1], dtype=np.float32)[None, :, None]
    rank /= max(neighbors.shape[1] - 1, 1)
    return np.concatenate(
        (
            np.broadcast_to(qpos[:, None], (*neighbors.shape, 2)),
            delta,
            unit,
            np.log1p(distances)[..., None],
            np.broadcast_to(rank, (*neighbors.shape, 1)),
            np.broadcast_to(qctx[:, None], (*neighbors.shape, 7)),
            cdelta,
        ),
        axis=-1,
    ).astype(np.float32)


def load_fold_expert_gate(
    path: Path | None, kind: str, fold: str, device: torch.device
) -> dict[str, object] | None:
    if path is None:
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    payload = checkpoint["fold_models"][kind][fold]
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
        "condition_mode": checkpoint.get("condition_mode", "basic"),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    data = RoundData(".")
    data.validate()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    suffix = args.checkpoint.stem.removeprefix("cv_noout_split")
    pas_gate = load_fold_expert_gate(
        args.disagreement_pas_model, "pas", suffix, device
    )
    pdp_gate = load_fold_expert_gate(
        args.disagreement_pdp_model, "pdp", suffix, device
    )
    pol_ue_scale = None
    if args.pol_ue_calibration is not None:
        calibration = json.loads(args.pol_ue_calibration.read_text(encoding="utf-8"))
        calibration_fold = next(row for row in calibration["loo"] if row["fold"] == suffix)
        pol_ue_scale = torch.complex(
            torch.tensor(calibration_fold["real"], dtype=torch.float32, device=device),
            torch.tensor(calibration_fold["imag"], dtype=torch.float32, device=device),
        )[None, :, None, :, None]
    all_positions = np.asarray(data.train_positions, dtype=np.float64)
    all_contexts = np.load(args.context)["train"].astype(np.float32)
    all_features = np.asarray(np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32)
    valid_global = nonzero_feature_indices(all_features)
    inverse = np.full(len(all_features), -1, dtype=np.int64)
    inverse[valid_global] = np.arange(len(valid_global))
    positions = all_positions[valid_global]
    contexts = all_contexts[valid_global]
    features = torch.from_numpy(all_features[valid_global]).to(device)
    val_global = np.asarray(checkpoint["validation_indices"], dtype=np.int64)[: args.limit]
    train_global = np.asarray(checkpoint["train_indices"], dtype=np.int64)
    val_idx, train_idx = inverse[val_global], inverse[train_global]
    embeddings = metric_embeddings(positions.astype(np.float32), contexts)
    configs = active_group_configs(args.advanced_map_expert)
    if args.advanced_map_expert and ADVANCED_MAP_METRIC not in embeddings:
        raise ValueError(
            f"--advanced-map-expert requires the 362-column advanced context; got {contexts.shape[1]}"
        )
    already_propagated = False
    if args.groupwise is None:
        pas = predict_config(PAS_CONFIG, embeddings[PAS_CONFIG.metric], train_idx, val_idx, features)
        pdp = predict_config(PDP_CONFIG, embeddings[PDP_CONFIG.metric], train_idx, val_idx, features)
        compact = torch.cat((pas[:, :1024], pdp[:, 1024:]), dim=1)
    else:
        groupwise = json.loads(args.groupwise.read_text(encoding="utf-8"))
        fold_meta = next(row for row in groupwise["loo"] if row["fold"] == suffix)
        group_weights = torch.tensor(fold_meta["weights"], device=device)
        bank = torch.stack(
            [
                predict_config(config, embeddings[config.metric], train_idx, val_idx, features)
                for config in configs
            ]
        )
        if args.conditional_groupwise is None:
            compact = mix_groups(bank, group_weights)
        else:
            conditional = json.loads(args.conditional_groupwise.read_text(encoding="utf-8"))
            regularization_key = f'{float(conditional["final"]["regularization"]):g}'
            fold_conditional = next(
                row
                for row in conditional["regularizations"][regularization_key]
                if row["fold"] == suffix
            )
            transition, boundary = graph_matrices(
                embeddings["xy_y0.75"], train_idx, val_idx, features,
                k=24, power=2.5, softening=0.0,
            )
            alpha = 0.1
            matrix = torch.eye(len(val_idx), device=device) - alpha * transition
            propagated = torch.linalg.solve(
                matrix,
                ((1.0 - alpha) * bank + alpha * boundary[None])
                .permute(1, 0, 2)
                .reshape(len(val_idx), -1),
            ).reshape(len(val_idx), len(configs), -1).permute(1, 0, 2)
            raw_condition = np.concatenate(
                (positions[val_idx, :2], contexts[val_idx, :7]), axis=1
            ).astype(np.float32)
            condition = torch.from_numpy(
                (raw_condition - np.asarray(fold_conditional["condition_mean"], dtype=np.float32))
                / np.asarray(fold_conditional["condition_std"], dtype=np.float32)
            ).to(device)
            corrections = torch.tensor(
                fold_conditional["pas_corrections"], dtype=torch.float32, device=device
            )
            pas_bank = propagated[:, :, :1024].reshape(len(configs), -1, 256, 4)
            pdp_bank = propagated[:, :, 1024:].reshape(len(configs), -1, 2, 4, 192)
            output_pas = torch.empty_like(pas_bank[0])
            output_pdp = torch.empty_like(pdp_bank[0])
            pas_logits = torch.stack(
                [
                    torch.log(group_weights[ue].clamp_min(1e-7))[None]
                    + condition @ corrections[ue]
                    for ue in range(4)
                ],
                dim=1,
            )
            pas_expert = pas_bank.permute(1, 3, 0, 2)
            if pas_gate is not None:
                raw_gate_condition = expert_gate_condition(
                    positions[val_idx], contexts[val_idx], pas_gate["condition_mode"]
                )
                gate_condition = (
                    torch.from_numpy(raw_gate_condition).to(device) - pas_gate["mean"]
                ) / pas_gate["std"]
                correction = torch.stack(
                    [model(gate_condition, pas_expert) for model in pas_gate["models"]]
                ).mean(0)
                pas_logits = pas_logits + pas_gate["scale"] * correction
            output_pas = torch.einsum(
                "quc,qucm->qmu", torch.softmax(pas_logits, dim=2), pas_expert
            )
            pdp_expert = pdp_bank.permute(1, 2, 3, 0, 4).reshape(
                len(val_idx), 8, len(configs), 192
            )
            pdp_logits = torch.log(group_weights[4:].clamp_min(1e-7))[None].expand(
                len(val_idx), -1, -1
            )
            if pdp_gate is not None:
                raw_gate_condition = expert_gate_condition(
                    positions[val_idx], contexts[val_idx], pdp_gate["condition_mode"]
                )
                gate_condition = (
                    torch.from_numpy(raw_gate_condition).to(device) - pdp_gate["mean"]
                ) / pdp_gate["std"]
                correction = torch.stack(
                    [model(gate_condition, pdp_expert) for model in pdp_gate["models"]]
                ).mean(0)
                pdp_logits = pdp_logits + pdp_gate["scale"] * correction
            output_pdp = torch.einsum(
                "qgc,qgcs->qgs", torch.softmax(pdp_logits, dim=2), pdp_expert
            ).reshape(len(val_idx), 2, 4, 192)
            compact = torch.cat((output_pas.flatten(1), output_pdp.flatten(1)), dim=1)
            already_propagated = True
    if not already_propagated:
        transition, boundary = graph_matrices(
            embeddings["xy_y0.75"], train_idx, val_idx, features,
            k=24, power=2.5, softening=0.0,
        )
        alpha = 0.1
        compact = torch.linalg.solve(
            torch.eye(len(val_idx), device=device) - alpha * transition,
            (1.0 - alpha) * compact + alpha * boundary,
        )
    if args.pas_pca_rank > 0 and args.pas_pca_blend > 0.0:
        compact_pas = compact[:, :1024].reshape(-1, 256, 4)
        train_pas = features[train_idx, :1024].reshape(-1, 256, 4)
        for ue in range(4):
            source = train_pas[:, :, ue]
            source = source / torch.linalg.vector_norm(source, dim=1, keepdim=True).clamp_min(1e-12)
            prediction = compact_pas[:, :, ue]
            prediction = prediction / torch.linalg.vector_norm(
                prediction, dim=1, keepdim=True
            ).clamp_min(1e-12)
            mean = source.mean(0, keepdim=True)
            centered = source - mean
            covariance = centered.T @ centered / len(source)
            _, vectors = torch.linalg.eigh(covariance)
            basis = vectors[:, -args.pas_pca_rank :]
            projected = mean + (prediction - mean) @ basis @ basis.T
            projected = projected.clamp_min(0.0)
            projected /= torch.linalg.vector_norm(projected, dim=1, keepdim=True).clamp_min(1e-12)
            compact_pas[:, :, ue] = (
                (1.0 - args.pas_pca_blend) * prediction
                + args.pas_pca_blend * projected
            ).clamp_min(0.0)
    if args.neural_blend > 0.0:
        state = checkpoint["state_dict"]
        model = MapConditionedKernel(
            state["position_mean"], state["position_std"], state["context_mean"], state["context_std"]
        ).to(device)
        model.load_state_dict(state)
        model.eval()
        neural_context_dim = int(state["context_mean"].numel())
        neural_local, neural_distance = nearest_neighbors(
            positions[val_idx], positions[train_idx], int(checkpoint["neighbors"])
        )
        neural_neighbor = train_idx[neural_local]
        with torch.inference_mode():
            pas_weight, pdp_weight = model(
                torch.from_numpy(positions[val_idx].astype(np.float32)).to(device),
                torch.from_numpy(contexts[val_idx, :neural_context_dim]).to(device),
                torch.from_numpy(positions[neural_neighbor].astype(np.float32)).to(device),
                torch.from_numpy(contexts[neural_neighbor, :neural_context_dim]).to(device),
            )
            neural = interpolate_features(features[neural_neighbor], pas_weight, pdp_weight, 1024)
        compact = compact.lerp(neural, args.neural_blend)
    if args.local_spectral_correction is not None:
        local_spectral_correction = LocalSpectralCorrection.load(
            args.local_spectral_correction
        )
        compact = local_spectral_correction.apply(
            compact, positions[val_idx], contexts[val_idx]
        )
    phase_coordinates = (
        positions if args.phase_neighbor_metric == "xy" else embeddings[args.phase_neighbor_metric]
    )
    neighbor_local, neighbor_distance = nearest_neighbors(
        phase_coordinates[val_idx], phase_coordinates[train_idx], args.phase_neighbors
    )
    neighbor_idx = train_idx[neighbor_local]
    learned_delaywise_model = None
    learned_delaywise_pair = None
    learned_delaywise_neighbors = 0
    if args.learned_delaywise_model is not None:
        if args.phase_neighbor_metric != "xy":
            raise ValueError("learned delay-wise attention requires --phase-neighbor-metric xy")
        learned_checkpoint = torch.load(
            args.learned_delaywise_model, map_location="cpu", weights_only=False
        )
        learned_delaywise_neighbors = int(learned_checkpoint["neighbors"])
        if learned_delaywise_neighbors > args.phase_neighbors:
            raise ValueError("attention model uses more neighbors than --phase-neighbors")
        learned_delaywise_model = DelaywiseNeighborAttention(
            int(learned_checkpoint["pair_features"])
        ).to(device)
        learned_delaywise_model.load_state_dict(learned_checkpoint["state_dict"])
        learned_delaywise_model.eval()
        learned_delaywise_pair = make_delaywise_pair_features(
            positions,
            contexts,
            val_idx,
            neighbor_idx[:, :learned_delaywise_neighbors],
            neighbor_distance[:, :learned_delaywise_neighbors],
            learned_checkpoint,
        )
    predicted_group_fraction = None
    if args.group_energy is not None:
        all_group_energy = np.load(args.group_energy)
        group_fraction = all_group_energy / np.maximum(
            all_group_energy.sum(axis=(1, 2), keepdims=True), 1e-30
        )
        def interpolate_group_fraction(metric: str) -> np.ndarray:
            coordinates = positions if metric == "xy" else embeddings[metric]
            local, distance = nearest_neighbors(
                coordinates[val_idx], coordinates[train_idx], args.group_energy_neighbors
            )
            weight = (distance + 1e-3) ** (-args.group_energy_power)
            weight /= weight.sum(axis=1, keepdims=True)
            prediction = np.exp(
                np.sum(
                    weight[:, :, None, None]
                    * np.log(
                        group_fraction[valid_global[train_idx[local]]].clip(1e-12)
                    ),
                    axis=1,
                )
            )
            return prediction / prediction.sum(axis=(1, 2), keepdims=True)

        predicted_group_fraction = interpolate_group_fraction(args.group_energy_metric)
        if args.group_energy_advanced_metric is not None:
            advanced_group_fraction = interpolate_group_fraction(
                args.group_energy_advanced_metric
            )
            use_advanced = (
                (neighbor_distance[:, 0] >= args.group_energy_advanced_min_distance)
                & (neighbor_distance[:, 0] < args.group_energy_advanced_max_distance)
            )
            predicted_group_fraction = np.where(
                use_advanced[:, None, None],
                advanced_group_fraction,
                predicted_group_fraction,
            )
    predicted_antenna_fraction = None
    predicted_subcarrier_fraction = None
    if args.power_marginals is not None:
        marginal_archive = np.load(args.power_marginals)
        energy_coordinates = (
            positions
            if args.group_energy_metric == "xy"
            else embeddings[args.group_energy_metric]
        )
        energy_local, energy_distance = nearest_neighbors(
            energy_coordinates[val_idx],
            energy_coordinates[train_idx],
            args.group_energy_neighbors,
        )
        energy_weight = (energy_distance + 1e-3) ** (-args.group_energy_power)
        energy_weight /= energy_weight.sum(axis=1, keepdims=True)
        energy_global = valid_global[train_idx[energy_local]]

        def interpolate_fraction(values: np.ndarray, normalization_axis: int) -> np.ndarray:
            fraction = values / np.maximum(
                values.sum(axis=normalization_axis, keepdims=True), 1e-30
            )
            prediction = np.exp(
                np.sum(
                    energy_weight[(...,) + (None,) * (fraction.ndim - 1)]
                    * np.log(fraction[energy_global].clip(1e-12)),
                    axis=1,
                )
            )
            return prediction / np.maximum(
                prediction.sum(axis=normalization_axis, keepdims=True), 1e-30
            )

        predicted_antenna_fraction = interpolate_fraction(
            marginal_archive["antenna_ue"], 1
        )
        predicted_subcarrier_fraction = interpolate_fraction(
            marginal_archive["ue_subcarrier"], 2
        )
    if args.phase_weighting == "kriging":
        neighbor_positions = phase_coordinates[neighbor_idx]
        pair = np.linalg.norm(
            neighbor_positions[:, :, None, :] - neighbor_positions[:, None, :, :], axis=-1
        )
        bandwidth = np.maximum(
            neighbor_distance[:, -1] * args.phase_kriging_scale, 1e-6
        )
        covariance_nn = np.exp(-pair / bandwidth[:, None, None])
        covariance_q = np.exp(-neighbor_distance / bandwidth[:, None])
        k = args.phase_neighbors
        system = np.zeros((len(val_idx), k + 1, k + 1), dtype=np.float64)
        system[:, :k, :k] = covariance_nn + np.eye(k)[None] * args.phase_kriging_nugget
        system[:, :k, k] = 1.0
        system[:, k, :k] = 1.0
        right = np.concatenate((covariance_q, np.ones((len(val_idx), 1))), axis=1)
        weights_np = np.linalg.solve(system, right[..., None])[..., 0][:, :k]
        weights_np = np.maximum(weights_np, 0.0)
        weights_np /= np.maximum(weights_np.sum(axis=1, keepdims=True), 1e-12)
        weights_np = weights_np.astype(np.float32)
    elif args.phase_softening == 0.0:
        weights_np = distance_weights(neighbor_distance, power=args.phase_power).astype(np.float32)
    else:
        weights_np = (neighbor_distance + args.phase_softening) ** (-args.phase_power)
        weights_np /= weights_np.sum(axis=1, keepdims=True)
        weights_np = weights_np.astype(np.float32)
    group_weights_np = None
    if args.phase_groupwise:
        if args.phase_neighbors != 16:
            raise ValueError("--phase-groupwise currently requires --phase-neighbors 16")
        neighbor_positions = phase_coordinates[neighbor_idx]
        pair = np.linalg.norm(
            neighbor_positions[:, :, None, :] - neighbor_positions[:, None, :, :], axis=-1
        )

        def group_kriging(neighbors: int, scale: float, nugget: float) -> np.ndarray:
            local_distance = neighbor_distance[:, :neighbors]
            local_pair = pair[:, :neighbors, :neighbors]
            bandwidth = np.maximum(local_distance[:, -1] * scale, 1e-6)
            covariance_nn = np.exp(-local_pair / bandwidth[:, None, None])
            covariance_q = np.exp(-local_distance / bandwidth[:, None])
            k = neighbors
            system = np.zeros((len(val_idx), k + 1, k + 1), dtype=np.float64)
            system[:, :k, :k] = covariance_nn + np.eye(k)[None] * nugget
            system[:, :k, k] = 1.0
            system[:, k, :k] = 1.0
            right = np.concatenate((covariance_q, np.ones((len(val_idx), 1))), axis=1)
            local_weight = np.linalg.solve(system, right[..., None])[..., 0][:, :k]
            weight = np.zeros((len(val_idx), args.phase_neighbors), dtype=np.float32)
            weight[:, :k] = local_weight
            return weight

        weak_weight = group_kriging(16, 2.0, 0.001)
        weak8_weight = group_kriging(8, 0.75, 0.001)
        strong_weight = group_kriging(16, 2.0, 0.1)
        strong_group = np.array(
            [[False, False, True, True], [True, True, False, False]]
        )
        group_weights_np = np.where(
            strong_group[None, None],
            strong_weight[:, :, None, None],
            weak_weight[:, :, None, None],
        )
        weak8_group = np.array(
            [[False, True, False, False], [False, False, False, True]]
        )
        if args.phase_weak8:
            group_weights_np = np.where(
                weak8_group[None, None],
                weak8_weight[:, :, None, None],
                group_weights_np,
            )
    bs = np.array([50.0, 0.0, 25.0])
    radius = np.linalg.norm(positions - bs, axis=1)
    radial_delta = radius[neighbor_idx] - radius[val_idx, None]
    direction = (positions - bs) / radius[:, None]
    direction_delta = direction[neighbor_idx] - direction[val_idx, None, :]
    totals = {
        str(k): {"pas": 0.0, "pdp": 0.0, "count": 0, "cross": 0j, "pred_energy": 0.0, "energy": 0.0}
        for k in args.wavenumbers
    }
    betas = (
        0.0,
        0.01,
        0.02,
        0.04,
        0.05,
        0.06,
        0.08,
        0.10,
        0.12,
        0.14,
        0.16,
        0.18,
        0.20,
        0.30,
        0.50,
        0.75,
        1.0,
    )
    blend_totals = {
        str(beta): {"pas": 0.0, "pdp": 0.0, "count": 0, "error": 0.0, "energy": 0.0}
        for beta in betas
    }
    energy_blend_totals = {
        str(beta): {"pas": 0.0, "pdp": 0.0, "count": 0, "cross": 0j, "pred_energy": 0.0, "energy": 0.0}
        for beta in betas
    }
    adaptive_edges = np.asarray(args.adaptive_edges, dtype=np.float64)
    if len(adaptive_edges) < 2 or np.any(np.diff(adaptive_edges) <= 0.0):
        raise ValueError("--adaptive-edges must be strictly increasing")
    _, physical_distance = nearest_neighbors(positions[val_idx], positions[train_idx], 1)
    adaptive_bin = np.digitize(physical_distance[:, 0], adaptive_edges[1:-1])
    adaptive_totals = {
        str(beta): [
            {"pas": 0.0, "pdp": 0.0, "count": 0, "cross": 0j, "pred_energy": 0.0, "energy": 0.0}
            for _ in range(len(adaptive_edges) - 1)
        ]
        for beta in betas
    }
    calibration_totals = {
        str(strength): {
            "pas": 0.0,
            "pdp": 0.0,
            "count": 0,
            "cross": 0j,
            "pred_energy": 0.0,
            "energy": 0.0,
        }
        for strength in args.pol_ue_strengths
    }
    group_energy_totals = {
        str(strength): {
            "pas": 0.0,
            "pdp": 0.0,
            "count": 0,
            "cross": 0j,
            "pred_energy": 0.0,
            "energy": 0.0,
        }
        for strength in args.group_energy_strengths
    }
    group_energy_bin_totals = {
        str(strength): [
            {
                "pas": 0.0, "pdp": 0.0, "count": 0, "cross": 0j,
                "pred_energy": 0.0, "energy": 0.0,
            }
            for _ in range(len(adaptive_edges) - 1)
        ]
        for strength in args.group_energy_strengths
    }
    hidw_phase_strengths = (0.0, 0.25, 0.5, 0.75, 1.0)
    hidw_amplitude_strengths = (0.0, 0.25, 0.5, 0.75, 1.0)
    hidw_alignment_totals = {
        f"g{group:g}_p{phase:g}_a{amplitude:g}": {
            "pas": 0.0, "pdp": 0.0, "count": 0, "cross": 0j,
            "pred_energy": 0.0, "energy": 0.0,
        }
        for group in args.group_energy_strengths
        for phase in hidw_phase_strengths
        for amplitude in hidw_amplitude_strengths
    }
    antenna_energy_totals = {
        str(strength): {
            "pas": 0.0, "pdp": 0.0, "count": 0, "cross": 0j,
            "pred_energy": 0.0, "energy": 0.0,
        }
        for strength in args.marginal_strengths
    }
    subcarrier_energy_totals = {
        str(strength): {
            "pas": 0.0, "pdp": 0.0, "count": 0, "cross": 0j,
            "pred_energy": 0.0, "energy": 0.0,
        }
        for strength in args.marginal_strengths
    }
    weak_blends = (0.0, 0.05, 0.1, 0.15, 0.2)
    strong_blends = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
    group_blend_totals = {
        f"w{weak:g}_s{strong:g}": {
            "pas": 0.0, "pdp": 0.0, "count": 0, "cross": 0j,
            "pred_energy": 0.0, "energy": 0.0,
        }
        for weak in weak_blends
        for strong in strong_blends
    }
    channels = data.train_channels
    sample_stats: dict[str, list[np.ndarray]] = {
        "global_index": [],
        "position": [],
        "context": [],
        "nearest_distance": [],
        "cross": [],
        "pred_energy": [],
        "target_energy": [],
        "final_cross": [],
        "final_pred_energy": [],
        "final_pas": [],
        "final_pdp": [],
        "final_cross_ue": [],
        "final_pred_energy_ue": [],
        "cross_ue": [],
        "pred_energy_ue": [],
        "target_energy_ue": [],
        "cross_pol_ue": [],
        "pred_energy_pol_ue": [],
        "target_energy_pol_ue": [],
        "cross_antenna_ue": [],
        "pred_energy_antenna_ue": [],
        "target_energy_antenna_ue": [],
        "cross_pol_ue_subcarrier": [],
        "pred_energy_pol_ue_subcarrier": [],
        "target_energy_pol_ue_subcarrier": [],
    }
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

    def accumulate_variant(
        item: dict[str, object],
        candidate: torch.Tensor,
        target: torch.Tensor,
        true_pas: torch.Tensor,
        true_pdp: torch.Tensor,
        batch: int,
    ) -> None:
        candidate_pas = pas_spectrum(candidate, data.dims)
        candidate_pdp = pdp_spectrum(candidate)
        item["pas"] += cosine_similarity_last(candidate_pas, true_pas).mean().item() * batch
        item["pdp"] += cosine_similarity_last(candidate_pdp, true_pdp).mean().item() * batch
        item["count"] += batch
        item["cross"] += torch.sum(torch.conj(candidate) * target).item()
        item["pred_energy"] += torch.sum(
            torch.abs(candidate).square(), dtype=torch.float64
        ).item()
        item["energy"] += torch.sum(torch.abs(target).square(), dtype=torch.float64).item()

    for start in range(0, len(val_idx), args.batch_size):
        stop = min(start + args.batch_size, len(val_idx))
        source = torch.from_numpy(
            np.array(channels[valid_global[neighbor_idx[start:stop]]], dtype=np.complex64, copy=True)
        ).to(device)
        target = torch.from_numpy(
            np.array(channels[val_global[start:stop]], dtype=np.complex64, copy=True)
        ).to(device)
        target_pas, target_pdp = spectral_targets_from_features(compact[start:stop], data.dims)
        for k in args.wavenumbers:
            hidw_reference = None
            delta = torch.from_numpy(radial_delta[start:stop].astype(np.float32)).to(device)
            phase = torch.exp(
                1j
                * delta[:, :, None]
                * (k + args.phase_slope * subcarrier_offset[None, None, :])
            )
            angular_delta = torch.from_numpy(
                direction_delta[start:stop].astype(np.float32)
            ).to(device)
            steering = torch.exp(
                1j
                * (
                    (
                        args.h_steering_x * angular_delta[:, :, 0, None]
                        + args.h_steering * angular_delta[:, :, 1, None]
                    )
                    * h_grid
                    + args.v_steering * angular_delta[:, :, 2, None] * v_grid
                )
            )
            if group_weights_np is None:
                coefficient = (
                    torch.from_numpy(weights_np[start:stop]).to(device)[:, :, None, None]
                    * steering[:, :, :, None]
                    * phase[:, :, None, :]
                )
                prediction = torch.sum(source * coefficient[:, :, :, None, :], dim=1)
            else:
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
                    * phase[:, :, None, None, None, :]
                )
                if args.evaluate_hidw_alignment:
                    hidw_neighbors = min(8, args.phase_neighbors)
                    hidw_distance = torch.from_numpy(
                        neighbor_distance[start:stop, :hidw_neighbors].astype(np.float32)
                    ).to(device)
                    hidw_spatial = hidw_distance.clamp_min(1e-6).pow(-2.0)
                    hidw_spatial /= hidw_spatial.sum(1, keepdim=True).clamp_min(1e-20)
                    _, hidw_reference = phase_aligned_idw(
                        adjusted_group[:, :hidw_neighbors].flatten(2, 3), hidw_spatial
                    )
                group_weight = torch.from_numpy(group_weights_np[start:stop]).to(device)
                if args.delay_sync_strength != 0.0:
                    delay_group = torch.fft.fft(adjusted_group, dim=-1, norm="ortho")
                    anchor = delay_group[:, :1]
                    delay_cross = torch.sum(
                        torch.conj(delay_group) * anchor, dim=3
                    )
                    delay_sync = delay_cross / torch.abs(delay_cross).clamp_min(1e-30)
                    delay_weight = group_weight[:, :, :, :, None] * torch.exp(
                        1j * args.delay_sync_strength * torch.angle(delay_sync)
                    )
                    prediction_delay = torch.sum(
                        delay_group * delay_weight[:, :, :, None, :, :], dim=1
                    )
                    prediction = torch.fft.ifft(
                        prediction_delay, dim=-1, norm="ortho"
                    ).reshape(stop - start, *data.dims.channel_shape)
                else:
                    if args.phase_sync_strength != 0.0:
                        gram = torch.einsum(
                            "bkpaus,blpaus->bpukl",
                            torch.conj(adjusted_group),
                            adjusted_group,
                        )
                        diagonal = torch.diagonal(gram, dim1=-2, dim2=-1).real.clamp_min(1e-30)
                        coherence = gram / torch.sqrt(
                            diagonal[..., :, None] * diagonal[..., None, :]
                        )
                        vectors = torch.linalg.eigh(coherence).eigenvectors[..., -1]
                        vectors *= torch.exp(-1j * torch.angle(vectors[..., :1]))
                        sync = vectors.permute(0, 3, 1, 2)
                        gram8 = gram[..., :8, :8]
                        diagonal8 = torch.diagonal(gram8, dim1=-2, dim2=-1).real.clamp_min(1e-30)
                        coherence8 = gram8 / torch.sqrt(
                            diagonal8[..., :, None] * diagonal8[..., None, :]
                        )
                        vectors8 = torch.linalg.eigh(coherence8).eigenvectors[..., -1]
                        vectors8 *= torch.exp(-1j * torch.angle(vectors8[..., :1]))
                        sync8 = torch.ones_like(sync)
                        sync8[:, :8] = vectors8.permute(0, 3, 1, 2)
                        weak8_mask = torch.tensor(
                            [[False, True, False, False], [False, False, False, True]],
                            device=device,
                        )
                        if args.phase_weak8:
                            sync = torch.where(weak8_mask[None, None], sync8, sync)
                        group_weight = group_weight * torch.exp(
                            1j * args.phase_sync_strength * torch.angle(sync)
                        )
                    prediction = torch.sum(
                        adjusted_group * group_weight[:, :, :, None, :, None], dim=1
                    ).reshape(
                        stop - start, *data.dims.channel_shape
                    )
            if args.delaywise_direct_strength > 0.0:
                if group_weights_np is None:
                    raise ValueError("--delaywise-direct-strength requires --phase-groupwise")
                delaywise = delaywise_attention_prediction(
                    adjusted_group,
                    torch.from_numpy(
                        neighbor_distance[start:stop].astype(np.float32)
                    ).to(device),
                    args.delaywise_neighbors,
                    args.delaywise_power,
                    args.delaywise_softening,
                    args.delaywise_coherence,
                    args.delaywise_energy,
                    args.delaywise_fusion,
                    args.delaywise_alignment,
                    args.delaywise_hidw_blend,
                    args.delaywise_angle_transform,
                    data.dims.bs_h,
                    data.dims.bs_v,
                )
                prediction = prediction.lerp(delaywise, args.delaywise_direct_strength)
            if learned_delaywise_model is not None and args.learned_delaywise_strength != 0.0:
                k_attention = learned_delaywise_neighbors
                learned_source = adjusted_group[:, :k_attention].flatten(2, 3)
                learned_distance = torch.from_numpy(
                    neighbor_distance[start:stop, :k_attention].astype(np.float32)
                ).to(device)
                learned_spatial = learned_distance.clamp_min(1e-6).pow(-2.0)
                learned_spatial /= learned_spatial.sum(1, keepdim=True).clamp_min(1e-20)
                learned_aligned, learned_centroid = phase_aligned_idw(
                    learned_source, learned_spatial
                )
                learned_coefficient = angle_delay_coefficients(learned_aligned)
                learned_coherence, learned_log_energy = observable_delay_statistics(
                    learned_coefficient
                )
                with torch.inference_mode():
                    learned_weight, _ = learned_delaywise_model(
                        torch.from_numpy(learned_delaywise_pair[start:stop]).to(device),
                        learned_coherence,
                        learned_log_energy,
                        torch.log(learned_spatial.clamp_min(1e-20)),
                    )
                learned_prediction, _ = reconstruct_from_attention(
                    learned_coefficient, learned_weight, learned_centroid, 0.1
                )
                prediction = prediction.lerp(
                    learned_prediction, args.learned_delaywise_strength
                )
            direct_prediction = prediction
            for _ in range(args.iterations):
                prediction = replace_magnitude(
                    prediction, target_pas, 1, args.relaxation
                )
                prediction = replace_magnitude(
                    prediction, target_pdp, -1, args.relaxation
                )
            if args.terminal_pdp > 0.0:
                prediction = replace_magnitude(
                    prediction, target_pdp, -1, args.terminal_pdp
                )
            prediction = refine_spectral_cosine(
                prediction,
                target_pas,
                target_pdp,
                data.dims,
                args.spectral_refine_steps,
                args.spectral_refine_lr,
                args.spectral_refine_anchor,
            )
            pred_pas = pas_spectrum(prediction, data.dims)
            true_pas = pas_spectrum(target, data.dims)
            pred_pdp = pdp_spectrum(prediction)
            true_pdp = pdp_spectrum(target)
            item = totals[str(k)]
            batch = stop - start
            item["pas"] += cosine_similarity_last(pred_pas, true_pas).mean().item() * batch
            item["pdp"] += cosine_similarity_last(pred_pdp, true_pdp).mean().item() * batch
            item["count"] += batch
            item["cross"] += torch.sum(torch.conj(prediction) * target).item()
            item["pred_energy"] += torch.sum(torch.abs(prediction).square(), dtype=torch.float64).item()
            item["energy"] += torch.sum(torch.abs(target).square(), dtype=torch.float64).item()
            if abs(k - args.blend_wavenumber) < 1e-9:
                ap_scale = complex(args.blend_scale_real, args.blend_scale_imag)
                energy_scale = torch.sqrt(
                    torch.sum(torch.abs(direct_prediction).square(), dim=(1, 2, 3), keepdim=True)
                    / torch.sum(torch.abs(prediction).square(), dim=(1, 2, 3), keepdim=True).clamp_min(1e-30)
                )
                source_energy = torch.sum(
                    torch.abs(source[:, : args.energy_neighbors]).square(),
                    dim=(2, 3, 4),
                    dtype=torch.float64,
                )
                geometric_energy = torch.exp(
                    torch.mean(torch.log(source_energy.clamp_min(1e-30)), dim=1)
                ).to(prediction.real.dtype)
                direct_energy = torch.sum(
                    torch.abs(direct_prediction).square(), dim=(1, 2, 3)
                ).clamp_min(1e-30)
                energy_correction = (geometric_energy / direct_energy).pow(
                    args.energy_gamma / 2.0
                )[:, None, None, None]
                for beta in betas:
                    blended = (1.0 - beta) * ap_scale * prediction + beta * direct_prediction
                    blended_pas = pas_spectrum(blended, data.dims)
                    blended_pdp = pdp_spectrum(blended)
                    blend_item = blend_totals[str(beta)]
                    blend_item["pas"] += cosine_similarity_last(blended_pas, true_pas).mean().item() * batch
                    blend_item["pdp"] += cosine_similarity_last(blended_pdp, true_pdp).mean().item() * batch
                    blend_item["count"] += batch
                    blend_item["error"] += torch.sum(
                        torch.abs(blended - target).square(), dtype=torch.float64
                    ).item()
                    blend_item["energy"] += torch.sum(
                        torch.abs(target).square(), dtype=torch.float64
                    ).item()
                    energy_blended = (
                        (1.0 - beta) * energy_scale * prediction + beta * direct_prediction
                    ) * energy_correction
                    if args.post_projection > 0.0:
                        projection_scale = energy_scale * energy_correction
                        normalized_blend = energy_blended / projection_scale.clamp_min(1e-30)
                        normalized_blend = replace_magnitude(
                            normalized_blend, target_pas, 1, args.post_projection
                        )
                        normalized_blend = replace_magnitude(
                            normalized_blend, target_pdp, -1, args.post_projection
                        )
                        energy_blended = normalized_blend * projection_scale
                    if abs(beta - args.diagnostic_beta) < 1e-9:
                        projected_group = (energy_scale * prediction).reshape(
                            batch,
                            data.dims.bs_polarizations,
                            data.dims.bs_h * data.dims.bs_v,
                            data.dims.ue_antennas,
                            data.dims.subcarriers,
                        )
                        direct_group = direct_prediction.reshape_as(projected_group)
                        strong_mask = torch.tensor(
                            [[False, False, True, True], [True, True, False, False]],
                            device=device,
                        )
                        for weak_blend in weak_blends:
                            for strong_blend in strong_blends:
                                group_beta = torch.where(
                                    strong_mask,
                                    torch.tensor(strong_blend, device=device),
                                    torch.tensor(weak_blend, device=device),
                                )[None, :, None, :, None]
                                group_blended = (
                                    (1.0 - group_beta) * projected_group
                                    + group_beta * direct_group
                                ) * energy_correction.reshape(batch, 1, 1, 1, 1)
                                accumulate_variant(
                                    group_blend_totals[
                                        f"w{weak_blend:g}_s{strong_blend:g}"
                                    ],
                                    group_blended.reshape_as(energy_blended),
                                    target,
                                    true_pas,
                                    true_pdp,
                                    batch,
                                )
                    if (
                        pol_ue_scale is not None
                        and abs(beta - args.diagnostic_beta) < 1e-9
                    ):
                        grouped = energy_blended.reshape(
                            batch,
                            data.dims.bs_polarizations,
                            data.dims.bs_h * data.dims.bs_v,
                            data.dims.ue_antennas,
                            data.dims.subcarriers,
                        )
                        for strength in args.pol_ue_strengths:
                            scale = 1.0 + strength * (pol_ue_scale - 1.0)
                            calibrated = (grouped * scale).reshape_as(energy_blended)
                            calibrated_pas = pas_spectrum(calibrated, data.dims)
                            calibrated_pdp = pdp_spectrum(calibrated)
                            calibration_item = calibration_totals[str(strength)]
                            calibration_item["pas"] += (
                                cosine_similarity_last(calibrated_pas, true_pas).mean().item()
                                * batch
                            )
                            calibration_item["pdp"] += (
                                cosine_similarity_last(calibrated_pdp, true_pdp).mean().item()
                                * batch
                            )
                            calibration_item["count"] += batch
                            calibration_item["cross"] += torch.sum(
                                torch.conj(calibrated) * target
                            ).item()
                            calibration_item["pred_energy"] += torch.sum(
                                torch.abs(calibrated).square(), dtype=torch.float64
                            ).item()
                            calibration_item["energy"] += torch.sum(
                                torch.abs(target).square(), dtype=torch.float64
                            ).item()
                    if (
                        predicted_group_fraction is not None
                        and abs(beta - args.diagnostic_beta) < 1e-9
                    ):
                        grouped = energy_blended.reshape(
                            batch,
                            data.dims.bs_polarizations,
                            data.dims.bs_h * data.dims.bs_v,
                            data.dims.ue_antennas,
                            data.dims.subcarriers,
                        )
                        current_group_energy = torch.sum(
                            torch.abs(grouped).square(), dim=(2, 4)
                        )
                        current_group_fraction = current_group_energy / current_group_energy.sum(
                            dim=(1, 2), keepdim=True
                        ).clamp_min(1e-30)
                        desired_group_fraction = torch.from_numpy(
                            predicted_group_fraction[start:stop].astype(np.float32)
                        ).to(device)
                        ratio = desired_group_fraction / current_group_fraction.clamp_min(1e-12)
                        for strength in args.group_energy_strengths:
                            group_scale = ratio.pow(strength / 2.0)[:, :, None, :, None]
                            calibrated = (grouped * group_scale).reshape_as(energy_blended)
                            calibrated_pas = pas_spectrum(calibrated, data.dims)
                            calibrated_pdp = pdp_spectrum(calibrated)
                            energy_group_item = group_energy_totals[str(strength)]
                            energy_group_item["pas"] += (
                                cosine_similarity_last(calibrated_pas, true_pas).mean().item()
                                * batch
                            )
                            energy_group_item["pdp"] += (
                                cosine_similarity_last(calibrated_pdp, true_pdp).mean().item()
                                * batch
                            )
                            energy_group_item["count"] += batch
                            energy_group_item["cross"] += torch.sum(
                                torch.conj(calibrated) * target
                            ).item()
                            energy_group_item["pred_energy"] += torch.sum(
                                torch.abs(calibrated).square(), dtype=torch.float64
                            ).item()
                            energy_group_item["energy"] += torch.sum(
                                torch.abs(target).square(), dtype=torch.float64
                            ).item()
                            sample_pas = cosine_similarity_last(
                                calibrated_pas, true_pas
                            ).flatten(1).mean(1)
                            sample_pdp = cosine_similarity_last(
                                calibrated_pdp, true_pdp
                            ).flatten(1).mean(1)
                            batch_bins = adaptive_bin[start:stop]
                            for bin_index in range(len(adaptive_edges) - 1):
                                mask_np = batch_bins == bin_index
                                if not np.any(mask_np):
                                    continue
                                mask = torch.from_numpy(mask_np).to(device)
                                bin_item = group_energy_bin_totals[str(strength)][bin_index]
                                bin_item["pas"] += sample_pas[mask].sum().item()
                                bin_item["pdp"] += sample_pdp[mask].sum().item()
                                bin_item["count"] += int(mask_np.sum())
                                bin_item["cross"] += torch.sum(
                                    torch.conj(calibrated[mask]) * target[mask]
                                ).item()
                                bin_item["pred_energy"] += torch.sum(
                                    torch.abs(calibrated[mask]).square(), dtype=torch.float64
                                ).item()
                                bin_item["energy"] += torch.sum(
                                    torch.abs(target[mask]).square(), dtype=torch.float64
                                ).item()
                            if args.evaluate_hidw_alignment:
                                reference_cross = torch.sum(
                                    torch.conj(calibrated) * hidw_reference,
                                    dim=(1, 2, 3),
                                )
                                calibrated_energy = torch.sum(
                                    torch.abs(calibrated).square(), dim=(1, 2, 3)
                                ).clamp_min(1e-30)
                                reference_energy = torch.sum(
                                    torch.abs(hidw_reference).square(), dim=(1, 2, 3)
                                ).clamp_min(1e-30)
                                pas_sum = (
                                    cosine_similarity_last(calibrated_pas, true_pas).mean().item()
                                    * batch
                                )
                                pdp_sum = (
                                    cosine_similarity_last(calibrated_pdp, true_pdp).mean().item()
                                    * batch
                                )
                                for phase_strength in hidw_phase_strengths:
                                    phase_factor = torch.exp(
                                        1j * phase_strength * torch.angle(reference_cross)
                                    )
                                    for amplitude_strength in hidw_amplitude_strengths:
                                        amplitude_factor = (
                                            reference_energy / calibrated_energy
                                        ).pow(amplitude_strength / 2.0)
                                        aligned = calibrated * (
                                            phase_factor * amplitude_factor
                                        )[:, None, None, None]
                                        alignment_item = hidw_alignment_totals[
                                            f"g{strength:g}_p{phase_strength:g}_a{amplitude_strength:g}"
                                        ]
                                        alignment_item["pas"] += pas_sum
                                        alignment_item["pdp"] += pdp_sum
                                        alignment_item["count"] += batch
                                        alignment_item["cross"] += torch.sum(
                                            torch.conj(aligned) * target
                                        ).item()
                                        alignment_item["pred_energy"] += torch.sum(
                                            torch.abs(aligned).square(), dtype=torch.float64
                                        ).item()
                                        alignment_item["energy"] += torch.sum(
                                            torch.abs(target).square(), dtype=torch.float64
                                        ).item()
                    if (
                        predicted_antenna_fraction is not None
                        and abs(beta - args.diagnostic_beta) < 1e-9
                    ):
                        current = torch.sum(torch.abs(energy_blended).square(), dim=3)
                        current /= current.sum(dim=1, keepdim=True).clamp_min(1e-30)
                        desired = torch.from_numpy(
                            predicted_antenna_fraction[start:stop].astype(np.float32)
                        ).to(device)
                        ratio = desired / current.clamp_min(1e-12)
                        for strength in args.marginal_strengths:
                            calibrated = energy_blended * ratio.pow(strength / 2.0)[..., None]
                            accumulate_variant(
                                antenna_energy_totals[str(strength)],
                                calibrated,
                                target,
                                true_pas,
                                true_pdp,
                                batch,
                            )
                    if (
                        predicted_subcarrier_fraction is not None
                        and abs(beta - args.diagnostic_beta) < 1e-9
                    ):
                        current = torch.sum(torch.abs(energy_blended).square(), dim=1)
                        current /= current.sum(dim=2, keepdim=True).clamp_min(1e-30)
                        desired = torch.from_numpy(
                            predicted_subcarrier_fraction[start:stop].astype(np.float32)
                        ).to(device)
                        ratio = desired / current.clamp_min(1e-12)
                        for strength in args.marginal_strengths:
                            calibrated = energy_blended * ratio.pow(strength / 2.0)[:, None]
                            accumulate_variant(
                                subcarrier_energy_totals[str(strength)],
                                calibrated,
                                target,
                                true_pas,
                                true_pdp,
                                batch,
                            )
                    if pol_ue_scale is not None:
                        energy_blended = (
                            energy_blended.reshape(
                                batch,
                                data.dims.bs_polarizations,
                                data.dims.bs_h * data.dims.bs_v,
                                data.dims.ue_antennas,
                                data.dims.subcarriers,
                            )
                            * pol_ue_scale
                        ).reshape_as(energy_blended)
                    energy_pas = pas_spectrum(energy_blended, data.dims)
                    energy_pdp = pdp_spectrum(energy_blended)
                    energy_item = energy_blend_totals[str(beta)]
                    energy_item["pas"] += cosine_similarity_last(energy_pas, true_pas).mean().item() * batch
                    energy_item["pdp"] += cosine_similarity_last(energy_pdp, true_pdp).mean().item() * batch
                    energy_item["count"] += batch
                    energy_item["cross"] += torch.sum(torch.conj(energy_blended) * target).item()
                    energy_item["pred_energy"] += torch.sum(
                        torch.abs(energy_blended).square(), dtype=torch.float64
                    ).item()
                    energy_item["energy"] += torch.sum(
                        torch.abs(target).square(), dtype=torch.float64
                    ).item()
                    if args.sample_stats is not None and abs(beta - args.diagnostic_beta) < 1e-9:
                        stats_prediction = energy_blended
                        if (
                            predicted_group_fraction is not None
                            and args.sample_stats_group_energy_strength is not None
                        ):
                            stats_grouped = energy_blended.reshape(
                                batch,
                                data.dims.bs_polarizations,
                                data.dims.bs_h * data.dims.bs_v,
                                data.dims.ue_antennas,
                                data.dims.subcarriers,
                            )
                            stats_current_energy = torch.sum(
                                torch.abs(stats_grouped).square(), dim=(2, 4)
                            )
                            stats_current_fraction = stats_current_energy / stats_current_energy.sum(
                                dim=(1, 2), keepdim=True
                            ).clamp_min(1e-30)
                            stats_desired_fraction = torch.from_numpy(
                                predicted_group_fraction[start:stop].astype(np.float32)
                            ).to(device)
                            stats_ratio = stats_desired_fraction / stats_current_fraction.clamp_min(1e-12)
                            stats_distance = physical_distance[start:stop, 0]
                            stats_strength = np.where(
                                (
                                    stats_distance
                                    >= args.sample_stats_group_energy_mid_min_distance
                                )
                                & (
                                    stats_distance
                                    < args.sample_stats_group_energy_mid_max_distance
                                ),
                                args.sample_stats_group_energy_mid_strength,
                                args.sample_stats_group_energy_strength,
                            ).astype(np.float32)
                            stats_scale = stats_ratio.pow(
                                torch.from_numpy(stats_strength).to(device)[:, None, None] / 2.0
                            )[:, :, None, :, None]
                            stats_prediction = (stats_grouped * stats_scale).reshape_as(
                                energy_blended
                            )
                        sample_stats["global_index"].append(val_global[start:stop].copy())
                        sample_stats["position"].append(positions[val_idx[start:stop]].copy())
                        sample_stats["context"].append(contexts[val_idx[start:stop]].copy())
                        sample_stats["nearest_distance"].append(
                            physical_distance[start:stop, 0].copy()
                        )
                        sample_stats["cross"].append(
                            torch.sum(
                                torch.conj(energy_blended) * target,
                                dim=(1, 2, 3),
                            ).detach().cpu().numpy()
                        )
                        sample_stats["pred_energy"].append(
                            torch.sum(
                                torch.abs(energy_blended).square(),
                                dim=(1, 2, 3),
                                dtype=torch.float64,
                            ).detach().cpu().numpy()
                        )
                        sample_stats["target_energy"].append(
                            torch.sum(
                                torch.abs(target).square(),
                                dim=(1, 2, 3),
                                dtype=torch.float64,
                            ).detach().cpu().numpy()
                        )
                        sample_stats["final_cross"].append(
                            torch.sum(
                                torch.conj(stats_prediction) * target,
                                dim=(1, 2, 3),
                            ).detach().cpu().numpy()
                        )
                        sample_stats["final_pred_energy"].append(
                            torch.sum(
                                torch.abs(stats_prediction).square(),
                                dim=(1, 2, 3),
                                dtype=torch.float64,
                            ).detach().cpu().numpy()
                        )
                        stats_pas = cosine_similarity_last(
                            pas_spectrum(stats_prediction, data.dims), true_pas
                        ).mean(dim=(1, 2))
                        stats_pdp = cosine_similarity_last(
                            pdp_spectrum(stats_prediction), true_pdp
                        ).mean(dim=(1, 2))
                        sample_stats["final_pas"].append(
                            stats_pas.detach().cpu().numpy()
                        )
                        sample_stats["final_pdp"].append(
                            stats_pdp.detach().cpu().numpy()
                        )
                        sample_stats["final_cross_ue"].append(
                            torch.sum(
                                torch.conj(stats_prediction) * target,
                                dim=(1, 3),
                            ).detach().cpu().numpy()
                        )
                        sample_stats["final_pred_energy_ue"].append(
                            torch.sum(
                                torch.abs(stats_prediction).square(), dim=(1, 3)
                            ).detach().cpu().numpy()
                        )
                        prediction_pol = energy_blended.reshape(
                            batch,
                            data.dims.bs_polarizations,
                            data.dims.bs_h * data.dims.bs_v,
                            data.dims.ue_antennas,
                            data.dims.subcarriers,
                        )
                        target_pol = target.reshape_as(prediction_pol)
                        sample_stats["cross_ue"].append(
                            torch.sum(
                                torch.conj(energy_blended) * target,
                                dim=(1, 3),
                            ).detach().cpu().numpy()
                        )
                        sample_stats["pred_energy_ue"].append(
                            torch.sum(
                                torch.abs(energy_blended).square(), dim=(1, 3)
                            ).detach().cpu().numpy()
                        )
                        sample_stats["target_energy_ue"].append(
                            torch.sum(torch.abs(target).square(), dim=(1, 3)).detach().cpu().numpy()
                        )
                        sample_stats["cross_pol_ue"].append(
                            torch.sum(
                                torch.conj(prediction_pol) * target_pol,
                                dim=(2, 4),
                            ).detach().cpu().numpy()
                        )
                        sample_stats["pred_energy_pol_ue"].append(
                            torch.sum(torch.abs(prediction_pol).square(), dim=(2, 4)).detach().cpu().numpy()
                        )
                        sample_stats["target_energy_pol_ue"].append(
                            torch.sum(torch.abs(target_pol).square(), dim=(2, 4)).detach().cpu().numpy()
                        )
                        sample_stats["cross_antenna_ue"].append(
                            torch.sum(
                                torch.conj(energy_blended) * target, dim=3
                            ).detach().cpu().numpy()
                        )
                        sample_stats["pred_energy_antenna_ue"].append(
                            torch.sum(torch.abs(energy_blended).square(), dim=3).detach().cpu().numpy()
                        )
                        sample_stats["target_energy_antenna_ue"].append(
                            torch.sum(torch.abs(target).square(), dim=3).detach().cpu().numpy()
                        )
                        sample_stats["cross_pol_ue_subcarrier"].append(
                            torch.sum(
                                torch.conj(prediction_pol) * target_pol, dim=2
                            ).detach().cpu().numpy()
                        )
                        sample_stats["pred_energy_pol_ue_subcarrier"].append(
                            torch.sum(torch.abs(prediction_pol).square(), dim=2).detach().cpu().numpy()
                        )
                        sample_stats["target_energy_pol_ue_subcarrier"].append(
                            torch.sum(torch.abs(target_pol).square(), dim=2).detach().cpu().numpy()
                        )
                    pas_each = cosine_similarity_last(energy_pas, true_pas).flatten(1).mean(1)
                    pdp_each = cosine_similarity_last(energy_pdp, true_pdp).flatten(1).mean(1)
                    local_bins = adaptive_bin[start:stop]
                    for bin_index in range(len(adaptive_edges) - 1):
                        selected_np = np.flatnonzero(local_bins == bin_index)
                        if len(selected_np) == 0:
                            continue
                        selected = torch.from_numpy(selected_np).to(device)
                        bin_item = adaptive_totals[str(beta)][bin_index]
                        bin_item["pas"] += pas_each[selected].sum().item()
                        bin_item["pdp"] += pdp_each[selected].sum().item()
                        bin_item["count"] += len(selected_np)
                        local_prediction = energy_blended[selected]
                        local_target = target[selected]
                        bin_item["cross"] += torch.sum(
                            torch.conj(local_prediction) * local_target
                        ).item()
                        bin_item["pred_energy"] += torch.sum(
                            torch.abs(local_prediction).square(), dtype=torch.float64
                        ).item()
                        bin_item["energy"] += torch.sum(
                            torch.abs(local_target).square(), dtype=torch.float64
                        ).item()
        print(f"processed {stop}/{len(val_idx)}", flush=True)
    result = {}
    for k, item in totals.items():
        c1 = item["pas"] / item["count"]
        c2 = item["pdp"] / item["count"]
        scale = item["cross"] / item["pred_energy"]
        nmse = 1.0 - abs(item["cross"]) ** 2 / (item["pred_energy"] * item["energy"])
        score = 0.4 * c1 + 0.4 * c2 + 0.2 / (1.0 + nmse)
        result[k] = {
            "c1_pas": c1,
            "c2_pdp": c2,
            "optimal_complex_scale": [scale.real, scale.imag],
            "c3_nmse": nmse,
            "score": score,
        }
    blends = {}
    for beta, item in blend_totals.items():
        if item["count"] == 0:
            continue
        c1 = item["pas"] / item["count"]
        c2 = item["pdp"] / item["count"]
        nmse = item["error"] / item["energy"]
        blends[beta] = {
            "c1_pas": c1,
            "c2_pdp": c2,
            "c3_nmse": nmse,
            "score": 0.4*c1+0.4*c2+0.2/(1.0+nmse),
        }
    energy_blends = {}
    for beta, item in energy_blend_totals.items():
        if item["count"] == 0:
            continue
        c1 = item["pas"] / item["count"]
        c2 = item["pdp"] / item["count"]
        scale = item["cross"] / item["pred_energy"]
        nmse = 1.0 - abs(item["cross"])**2 / (item["pred_energy"] * item["energy"])
        energy_blends[beta] = {
            "c1_pas": c1,
            "c2_pdp": c2,
            "optimal_complex_scale": [scale.real, scale.imag],
            "c3_nmse": nmse,
            "score": 0.4*c1+0.4*c2+0.2/(1.0+nmse),
        }
    adaptive = {}
    for beta, bins in adaptive_totals.items():
        adaptive[beta] = []
        for item in bins:
            adaptive[beta].append(
                {
                    "pas": item["pas"],
                    "pdp": item["pdp"],
                    "count": item["count"],
                    "cross": [item["cross"].real, item["cross"].imag],
                    "pred_energy": item["pred_energy"],
                    "energy": item["energy"],
                }
            )
    calibration_results = {}
    for strength, item in calibration_totals.items():
        if item["count"] == 0:
            continue
        c1 = item["pas"] / item["count"]
        c2 = item["pdp"] / item["count"]
        scale = item["cross"] / item["pred_energy"]
        nmse = 1.0 - abs(item["cross"]) ** 2 / (
            item["pred_energy"] * item["energy"]
        )
        calibration_results[strength] = {
            "c1_pas": c1,
            "c2_pdp": c2,
            "optimal_complex_scale": [scale.real, scale.imag],
            "c3_nmse": nmse,
            "score": 0.4 * c1 + 0.4 * c2 + 0.2 / (1.0 + nmse),
        }
    group_energy_results = {}
    for strength, item in group_energy_totals.items():
        if item["count"] == 0:
            continue
        c1 = item["pas"] / item["count"]
        c2 = item["pdp"] / item["count"]
        scale = item["cross"] / item["pred_energy"]
        nmse = 1.0 - abs(item["cross"]) ** 2 / (
            item["pred_energy"] * item["energy"]
        )
        group_energy_results[strength] = {
            "c1_pas": c1,
            "c2_pdp": c2,
            "optimal_complex_scale": [scale.real, scale.imag],
            "c3_nmse": nmse,
            "score": 0.4 * c1 + 0.4 * c2 + 0.2 / (1.0 + nmse),
        }
    group_energy_bins = {
        strength: [
            {
                "pas": item["pas"],
                "pdp": item["pdp"],
                "count": item["count"],
                "cross": [item["cross"].real, item["cross"].imag],
                "pred_energy": item["pred_energy"],
                "energy": item["energy"],
            }
            for item in bins
        ]
        for strength, bins in group_energy_bin_totals.items()
    }

    def summarize_variants(totals_by_strength: dict[str, dict[str, object]]) -> dict:
        summary = {}
        for strength, item in totals_by_strength.items():
            if item["count"] == 0:
                continue
            c1 = item["pas"] / item["count"]
            c2 = item["pdp"] / item["count"]
            scale = item["cross"] / item["pred_energy"]
            nmse = 1.0 - abs(item["cross"]) ** 2 / (
                item["pred_energy"] * item["energy"]
            )
            summary[strength] = {
                "c1_pas": c1,
                "c2_pdp": c2,
                "optimal_complex_scale": [scale.real, scale.imag],
                "c3_nmse": nmse,
                "score": 0.4 * c1 + 0.4 * c2 + 0.2 / (1.0 + nmse),
            }
        return summary

    output = {
        "context": str(args.context),
        "advanced_map_expert": args.advanced_map_expert,
        "expert_metrics": [config.metric for config in configs],
        "phase": result,
        "blends": blends,
        "energy_blends": energy_blends,
        "adaptive_edges": adaptive_edges.tolist(),
        "adaptive_bins": adaptive,
        "pol_ue_strengths": calibration_results,
        "group_energy_strengths": group_energy_results,
        "group_energy_bins": group_energy_bins,
        "hidw_alignment": summarize_variants(hidw_alignment_totals),
        "antenna_energy_strengths": summarize_variants(antenna_energy_totals),
        "subcarrier_energy_strengths": summarize_variants(subcarrier_energy_totals),
        "group_blends": summarize_variants(group_blend_totals),
    }
    print(json.dumps(output, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if args.sample_stats is not None:
        args.sample_stats.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.sample_stats,
            **{name: np.concatenate(values, axis=0) for name, values in sample_stats.items()},
        )


if __name__ == "__main__":
    main()
