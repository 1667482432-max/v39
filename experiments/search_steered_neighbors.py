from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from physical_ai.data import RoundData
from physical_ai.features import nonzero_feature_indices
from physical_ai.neighbors import nearest_neighbors
from experiments.search_spatial_kernels_gpu import metric_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search KNN weights after full physical steering")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/cv_noout_split20260804.pt"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-neighbors", type=int, default=32)
    parser.add_argument("--neighbor-metric", type=str, default="xy")
    parser.add_argument("--statistics", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/steered_neighbor_search.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = RoundData(".")
    data.validate()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    features = np.asarray(np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32)
    valid = nonzero_feature_indices(features)
    inverse = np.full(len(features), -1, dtype=np.int64)
    inverse[valid] = np.arange(len(valid))
    positions = np.asarray(data.train_positions, dtype=np.float64)[valid]
    contexts = np.load("artifacts/map_context.npz")["train"].astype(np.float32)[valid]
    val_global = np.asarray(checkpoint["validation_indices"], dtype=np.int64)[: args.limit]
    train_global = np.asarray(checkpoint["train_indices"], dtype=np.int64)
    val_idx, train_idx = inverse[val_global], inverse[train_global]
    if args.neighbor_metric == "xy":
        search_coordinates = positions[:, :2]
    else:
        search_coordinates = metric_embeddings(positions.astype(np.float32), contexts)[args.neighbor_metric]
    local, distances = nearest_neighbors(
        search_coordinates[val_idx], search_coordinates[train_idx], args.max_neighbors
    )
    neighbors = train_idx[local]
    bs = np.asarray(data.dims.bs_position, dtype=np.float64)
    offset = positions - bs
    radius = np.linalg.norm(offset, axis=1)
    direction = offset / radius[:, None]
    radial_delta = radius[neighbors] - radius[val_idx, None]
    direction_delta = direction[neighbors] - direction[val_idx, None, :]
    q_count, kmax = neighbors.shape

    device = torch.device("cuda")
    frequency = torch.arange(data.dims.subcarriers, device=device, dtype=torch.float32)
    frequency -= (data.dims.subcarriers - 1) / 2.0
    h = torch.arange(data.dims.bs_h, device=device, dtype=torch.float32)
    h -= (data.dims.bs_h - 1) / 2.0
    v = torch.arange(data.dims.bs_v, device=device, dtype=torch.float32)
    v -= (data.dims.bs_v - 1) / 2.0
    h_grid = h[:, None].expand(data.dims.bs_h, data.dims.bs_v).reshape(-1).repeat(data.dims.bs_polarizations)
    v_grid = v[None, :].expand(data.dims.bs_h, data.dims.bs_v).reshape(-1).repeat(data.dims.bs_polarizations)
    cross = np.empty((q_count, kmax), dtype=np.complex128)
    gram = np.empty((q_count, kmax, kmax), dtype=np.complex128)
    target_energy = np.empty(q_count, dtype=np.float64)
    source_energy = np.empty((q_count, kmax), dtype=np.float64)
    group_cross = np.empty(
        (q_count, kmax, data.dims.bs_polarizations, data.dims.ue_antennas),
        dtype=np.complex128,
    )
    group_gram = np.empty(
        (
            q_count,
            kmax,
            kmax,
            data.dims.bs_polarizations,
            data.dims.ue_antennas,
        ),
        dtype=np.complex128,
    )
    group_target_energy = np.empty(
        (q_count, data.dims.bs_polarizations, data.dims.ue_antennas),
        dtype=np.float64,
    )
    channels = data.train_channels
    for row in range(q_count):
        source = torch.from_numpy(np.array(channels[valid[neighbors[row]]], copy=True)).to(device)
        target = torch.from_numpy(np.array(channels[val_global[row]], copy=True)).to(device)
        radial = torch.from_numpy(radial_delta[row].astype(np.float32)).to(device)
        angular = torch.from_numpy(direction_delta[row].astype(np.float32)).to(device)
        radial_phase = torch.exp(1j * radial[:, None] * (140.25 + 0.0006 * frequency))
        steering = torch.exp(
            1j
            * (
                (-1.75 * angular[:, 0, None] - 2.5 * angular[:, 1, None]) * h_grid
                + 26.0 * angular[:, 2, None] * v_grid
            )
        )
        adjusted = source * steering[:, :, None, None] * radial_phase[:, None, None, :]
        flat = adjusted.reshape(kmax, -1)
        target_flat = target.reshape(-1)
        cross[row] = (torch.conj(flat) @ target_flat).cpu().numpy().astype(np.complex128)
        gram[row] = (torch.conj(flat) @ flat.T).cpu().numpy().astype(np.complex128)
        target_energy[row] = torch.sum(torch.abs(target).square(), dtype=torch.float64).item()
        source_energy[row] = torch.sum(torch.abs(source).square(), dim=(1, 2, 3), dtype=torch.float64).cpu().numpy()
        adjusted_group = adjusted.reshape(
            kmax,
            data.dims.bs_polarizations,
            data.dims.bs_h * data.dims.bs_v,
            data.dims.ue_antennas,
            data.dims.subcarriers,
        )
        target_group = target.reshape(
            data.dims.bs_polarizations,
            data.dims.bs_h * data.dims.bs_v,
            data.dims.ue_antennas,
            data.dims.subcarriers,
        )
        group_cross[row] = torch.sum(
            torch.conj(adjusted_group) * target_group[None], dim=(2, 4)
        ).cpu().numpy()
        group_gram[row] = torch.einsum(
            "kpaus,lpaus->klpu", torch.conj(adjusted_group), adjusted_group
        ).cpu().numpy()
        group_target_energy[row] = torch.sum(
            torch.abs(target_group).square(), dim=(1, 3), dtype=torch.float64
        ).cpu().numpy()
        if (row + 1) % 20 == 0:
            print(f"correlations {row + 1}/{q_count}", flush=True)

    results = {}
    total_target_energy = target_energy.sum()
    def evaluate_weight(name: str, weight: np.ndarray) -> None:
        total_cross = np.einsum("qk,qk->", np.conj(weight), cross[:, : weight.shape[1]], optimize=True)
        prediction_energy = np.einsum(
            "qk,qkl,ql->",
            np.conj(weight),
            gram[:, : weight.shape[1], : weight.shape[1]],
            weight,
            optimize=True,
        ).real
        rho2 = abs(total_cross) ** 2 / max(prediction_energy * total_target_energy, 1e-30)
        local_group_cross = np.einsum(
            "qk,qkpu->pu", np.conj(weight), group_cross[:, : weight.shape[1]], optimize=True
        )
        local_group_energy = np.einsum(
            "qk,qklpu,ql->pu",
            np.conj(weight),
            group_gram[:, : weight.shape[1], : weight.shape[1]],
            weight,
            optimize=True,
        ).real
        group_rho2 = np.abs(local_group_cross) ** 2 / np.maximum(
            local_group_energy * group_target_energy.sum(axis=0), 1e-30
        )
        results[name] = {
            "rho_squared": float(rho2),
            "nmse": float(1.0 - rho2),
            "group_nmse": (1.0 - group_rho2).tolist(),
        }

    for k in (1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32):
        if k > kmax:
            continue
        for power in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0):
            for softening in (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0):
                weight = np.maximum(distances[:, :k] + softening, 1e-6) ** (-power)
                weight /= weight.sum(axis=1, keepdims=True)
                name = f"k{k}_p{power:g}_e{softening:g}"
                evaluate_weight(name, weight)
    # Smooth residual-channel kernels after carrier and array steering.
    for k in (4, 6, 8, 10, 12, 16, 24, 32):
        local_distance = distances[:, :k]
        bandwidth = np.maximum(local_distance[:, -1:], 1e-6)
        for scale in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
            gaussian = np.exp(-0.5 * (local_distance / (bandwidth * scale)) ** 2)
            gaussian /= gaussian.sum(axis=1, keepdims=True)
            evaluate_weight(f"gauss_k{k}_s{scale:g}", gaussian)
            exponential = np.exp(-local_distance / (bandwidth * scale))
            exponential /= exponential.sum(axis=1, keepdims=True)
            evaluate_weight(f"exp_k{k}_s{scale:g}", exponential)
    query_xy = search_coordinates[val_idx]
    neighbor_xy = search_coordinates[neighbors]
    for k in (4, 6, 8, 10, 12, 16):
        pair = np.linalg.norm(
            neighbor_xy[:, :k, None, :] - neighbor_xy[:, None, :k, :], axis=-1
        )
        query_distance = distances[:, :k]
        bandwidth = np.maximum(query_distance[:, -1], 1e-6)
        for scale in (0.5, 0.75, 1.0, 1.5, 2.0):
            for nugget in (0.001, 0.01, 0.05, 0.1):
                normalized_pair = pair / (bandwidth[:, None, None] * scale)
                normalized_query = query_distance / (bandwidth[:, None] * scale)
                covariance_nn = np.exp(-normalized_pair)
                covariance_q = np.exp(-normalized_query)
                system = np.zeros((q_count, k + 1, k + 1), dtype=np.float64)
                system[:, :k, :k] = covariance_nn
                system[:, :k, :k] += np.eye(k)[None] * nugget
                system[:, :k, k] = 1.0
                system[:, k, :k] = 1.0
                right = np.concatenate((covariance_q, np.ones((q_count, 1))), axis=1)
                weight = np.linalg.solve(system, right[..., None])[..., 0][:, :k]
                evaluate_weight(f"ok_exp_k{k}_s{scale:g}_n{nugget:g}", weight)
                positive = np.maximum(weight, 0.0)
                positive /= np.maximum(positive.sum(axis=1, keepdims=True), 1e-12)
                evaluate_weight(f"okp_exp_k{k}_s{scale:g}_n{nugget:g}", positive)
    top = sorted(results.items(), key=lambda item: item[1]["nmse"])[:100]
    diagnostics = {}
    for name, k, power, softening in (
        ("baseline", 12, 4.0, 2.0),
        ("robust", 10, 3.0, 1.0),
    ):
        weight = np.maximum(distances[:, :k] + softening, 1e-6) ** (-power)
        weight /= weight.sum(axis=1, keepdims=True)
        per_cross = np.einsum("qk,qk->q", np.conj(weight), cross[:, :k], optimize=True)
        per_prediction_energy = np.einsum(
            "qk,qkl,ql->q", np.conj(weight), gram[:, :k, :k], weight, optimize=True
        ).real
        diagnostics[name] = {
            "cross_real": per_cross.real.tolist(),
            "cross_imag": per_cross.imag.tolist(),
            "prediction_energy": per_prediction_energy.tolist(),
        }
    output = {
        "top": top,
        "results": results,
        "baseline": results["k12_p4_e2"],
        "target_energy": target_energy.tolist(),
        "source_energy": source_energy.tolist(),
        "distances": distances.tolist(),
        "diagnostics": diagnostics,
    }
    print(json.dumps({"top": top[:20], "baseline": output["baseline"]}, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output), encoding="utf-8")
    if args.statistics is not None:
        args.statistics.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.statistics,
            cross=cross,
            gram=gram,
            group_cross=group_cross,
            group_gram=group_gram,
            target_energy=target_energy,
            group_target_energy=group_target_energy,
            source_energy=source_energy,
            distances=distances,
            validation_indices=val_global,
            neighbor_indices=valid[neighbors],
        )


if __name__ == "__main__":
    main()
