from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from physical_ai.data import RoundData
from physical_ai.features import nonzero_feature_indices
from physical_ai.neighbors import nearest_neighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search array steering phase compensation")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/cv_noout_split20260804.pt"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--max-coefficient", type=float, default=6.0)
    parser.add_argument("--kriging", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/antenna_steering.json"))
    return parser.parse_args()


def quality(
    coefficient: torch.Tensor,
    cross: torch.Tensor,
    gram: torch.Tensor,
    target_energy: torch.Tensor,
) -> float:
    total_cross = torch.einsum("qkm,qkm->", torch.conj(coefficient), cross)
    prediction_energy = torch.einsum(
        "qkm,qklm,qlm->", torch.conj(coefficient), gram, coefficient
    ).real
    return float((torch.abs(total_cross).square() / (prediction_energy * target_energy.sum())).item())


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
    val_global = np.asarray(checkpoint["validation_indices"], dtype=np.int64)[: args.limit]
    train_global = np.asarray(checkpoint["train_indices"], dtype=np.int64)
    val_idx, train_idx = inverse[val_global], inverse[train_global]
    neighbor_count = 16 if args.kriging else 12
    local, distance = nearest_neighbors(positions[val_idx], positions[train_idx], neighbor_count)
    neighbor = train_idx[local]
    if args.kriging:
        neighbor_xy = positions[neighbor, :2]
        pair = np.linalg.norm(
            neighbor_xy[:, :, None, :] - neighbor_xy[:, None, :, :], axis=-1
        )
        bandwidth = np.maximum(distance[:, -1] * 0.75, 1e-6)
        cnn = np.exp(-pair / bandwidth[:, None, None])
        cq = np.exp(-distance / bandwidth[:, None])
        system = np.zeros((len(val_idx), neighbor_count + 1, neighbor_count + 1))
        system[:, :neighbor_count, :neighbor_count] = cnn + np.eye(neighbor_count)[None] * 0.1
        system[:, :neighbor_count, neighbor_count] = 1.0
        system[:, neighbor_count, :neighbor_count] = 1.0
        right = np.concatenate((cq, np.ones((len(val_idx), 1))), axis=1)
        weight = np.linalg.solve(system, right[..., None])[..., 0][:, :neighbor_count]
        weight = np.maximum(weight, 0.0)
        weight /= weight.sum(axis=1, keepdims=True)
    else:
        weight = (distance + 2.0) ** -4.0
        weight /= weight.sum(axis=1, keepdims=True)
    bs = np.asarray(data.dims.bs_position, dtype=np.float64)
    offset = positions - bs
    radius = np.linalg.norm(offset, axis=1)
    direction = offset / radius[:, None]
    radial_delta = radius[neighbor] - radius[val_idx, None]
    direction_delta = direction[neighbor] - direction[val_idx, None, :]

    device = torch.device("cuda")
    channels = data.train_channels
    q_count, k_count, antennas = len(val_idx), neighbor_count, data.dims.bs_antennas
    cross = torch.empty((q_count, k_count, antennas), dtype=torch.complex64, device=device)
    gram = torch.empty((q_count, k_count, k_count, antennas), dtype=torch.complex64, device=device)
    target_energy = torch.empty(q_count, dtype=torch.float64, device=device)
    for q in range(q_count):
        source = torch.from_numpy(np.array(channels[valid[neighbor[q]]], copy=True)).to(device)
        target = torch.from_numpy(np.array(channels[val_global[q]], copy=True)).to(device)
        cross[q] = torch.einsum("kmns,mns->km", torch.conj(source), target)
        gram[q] = torch.einsum("kmns,lmns->klm", torch.conj(source), source)
        target_energy[q] = torch.sum(torch.abs(target).square(), dtype=torch.float64)
        if (q + 1) % 25 == 0:
            print(f"correlations {q + 1}/{q_count}", flush=True)

    base = torch.from_numpy(
        (weight * np.exp(1j * 140.25 * radial_delta)).astype(np.complex64)
    ).to(device)[:, :, None]
    h_index = torch.arange(data.dims.bs_h, device=device, dtype=torch.float32)
    h_index -= (data.dims.bs_h - 1) / 2.0
    v_index = torch.arange(data.dims.bs_v, device=device, dtype=torch.float32)
    v_index -= (data.dims.bs_v - 1) / 2.0
    h_grid = h_index[:, None].expand(data.dims.bs_h, data.dims.bs_v).reshape(-1)
    v_grid = v_index[None, :].expand(data.dims.bs_h, data.dims.bs_v).reshape(-1)
    h_grid = h_grid.repeat(data.dims.bs_polarizations)
    v_grid = v_grid.repeat(data.dims.bs_polarizations)
    delta = torch.from_numpy(direction_delta.astype(np.float32)).to(device)
    candidates = np.arange(-args.max_coefficient, args.max_coefficient + args.step * 0.5, args.step)

    scans: dict[str, dict[str, float]] = {}
    best_axis, best_coefficient, best_rho2 = "none", 0.0, quality(
        base.expand(-1, -1, antennas), cross, gram, target_energy
    )
    scans["baseline"] = {"coefficient": 0.0, "rho_squared": best_rho2}
    for axis_name, axis in (("h_ux", 0), ("h_uy", 1), ("h_uz", 2)):
        local_best = (-1.0, 0.0)
        for candidate in candidates:
            phase = torch.exp(1j * float(candidate) * delta[:, :, axis, None] * h_grid)
            rho2 = quality(base * phase, cross, gram, target_energy)
            if rho2 > local_best[0]:
                local_best = rho2, float(candidate)
        scans[axis_name] = {"coefficient": local_best[1], "rho_squared": local_best[0]}
        if local_best[0] > best_rho2:
            best_axis, best_coefficient, best_rho2 = axis_name, local_best[1], local_best[0]

    best_axis_index = {"h_ux": 0, "h_uy": 1, "h_uz": 2}.get(best_axis, 0)
    horizontal = torch.exp(
        1j * best_coefficient * delta[:, :, best_axis_index, None] * h_grid
    )
    for axis_name, axis in (("v_ux", 0), ("v_uy", 1), ("v_uz", 2)):
        local_best = (-1.0, 0.0)
        for candidate in candidates:
            phase = horizontal * torch.exp(
                1j * float(candidate) * delta[:, :, axis, None] * v_grid
            )
            rho2 = quality(base * phase, cross, gram, target_energy)
            if rho2 > local_best[0]:
                local_best = rho2, float(candidate)
        scans[f"{best_axis}+{axis_name}"] = {
            "h_coefficient": best_coefficient,
            "v_coefficient": local_best[1],
            "rho_squared": local_best[0],
        }

    # A planar array may be rotated relative to the map axes. Search the two
    # horizontal direction-cosine coefficients jointly while holding the very
    # stable vertical coefficient at its cross-fold value.
    horizontal_candidates = np.arange(-6.0, 6.0 + 0.25, 0.5)
    joint_best = (-1.0, 0.0, 0.0)
    vertical_phase = 26.0 * delta[:, :, 2, None] * v_grid
    for coefficient_x in horizontal_candidates:
        for coefficient_y in horizontal_candidates:
            steering = torch.exp(
                1j
                * (
                    float(coefficient_x) * delta[:, :, 0, None] * h_grid
                    + float(coefficient_y) * delta[:, :, 1, None] * h_grid
                    + vertical_phase
                )
            )
            rho2 = quality(base * steering, cross, gram, target_energy)
            if rho2 > joint_best[0]:
                joint_best = rho2, float(coefficient_x), float(coefficient_y)
    scans["joint_hxy+v_uz26"] = {
        "h_x_coefficient": joint_best[1],
        "h_y_coefficient": joint_best[2],
        "v_coefficient": 26.0,
        "rho_squared": joint_best[0],
    }

    result = {"scans": scans}
    print(json.dumps(result, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
