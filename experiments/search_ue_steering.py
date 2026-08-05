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
    parser = argparse.ArgumentParser(description="Search UE-array steering after BS steering")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/cv_noout_split20260804.pt"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ue_steering.json"))
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
    val_global = np.asarray(checkpoint["validation_indices"], dtype=np.int64)[: args.limit]
    train_global = np.asarray(checkpoint["train_indices"], dtype=np.int64)
    val_idx, train_idx = inverse[val_global], inverse[train_global]
    local, distance = nearest_neighbors(positions[val_idx], positions[train_idx], 12)
    neighbor = train_idx[local]
    weight = (distance + 2.0) ** -4.0
    weight /= weight.sum(axis=1, keepdims=True)
    bs = np.asarray(data.dims.bs_position, dtype=np.float64)
    offset = positions - bs
    radius = np.linalg.norm(offset, axis=1)
    direction = offset / radius[:, None]
    radial_delta = radius[neighbor] - radius[val_idx, None]
    direction_delta = direction[neighbor] - direction[val_idx, None, :]

    device = torch.device("cuda")
    h = torch.arange(data.dims.bs_h, device=device, dtype=torch.float32)
    h -= (data.dims.bs_h - 1) / 2.0
    v = torch.arange(data.dims.bs_v, device=device, dtype=torch.float32)
    v -= (data.dims.bs_v - 1) / 2.0
    h_grid = h[:, None].expand(data.dims.bs_h, data.dims.bs_v).reshape(-1).repeat(data.dims.bs_polarizations)
    v_grid = v[None, :].expand(data.dims.bs_h, data.dims.bs_v).reshape(-1).repeat(data.dims.bs_polarizations)
    cross = torch.empty((len(val_idx), 12, data.dims.ue_antennas), dtype=torch.complex64, device=device)
    gram = torch.empty((len(val_idx), 12, 12, data.dims.ue_antennas), dtype=torch.complex64, device=device)
    target_energy = torch.empty(len(val_idx), dtype=torch.float64, device=device)
    channels = data.train_channels
    for q in range(len(val_idx)):
        source = torch.from_numpy(np.array(channels[valid[neighbor[q]]], copy=True)).to(device)
        target = torch.from_numpy(np.array(channels[val_global[q]], copy=True)).to(device)
        delta = torch.from_numpy(direction_delta[q].astype(np.float32)).to(device)
        base = torch.from_numpy(
            (weight[q] * np.exp(1j * 140.25 * radial_delta[q])).astype(np.complex64)
        ).to(device)
        steering = torch.exp(
            1j
            * (
                (-1.75 * delta[:, 0, None] - 2.5 * delta[:, 1, None]) * h_grid
                + 26.0 * delta[:, 2, None] * v_grid
            )
        )
        adjusted = source * base[:, None, None, None] * steering[:, :, None, None]
        cross[q] = torch.einsum("kmns,mns->kn", torch.conj(adjusted), target)
        gram[q] = torch.einsum("kmns,lmns->kln", torch.conj(adjusted), adjusted)
        target_energy[q] = torch.sum(torch.abs(target).square(), dtype=torch.float64)
        if (q + 1) % 25 == 0:
            print(f"correlations {q + 1}/{len(val_idx)}", flush=True)

    delta = torch.from_numpy(direction_delta.astype(np.float32)).to(device)
    ue_v = torch.arange(data.dims.ue_v, device=device, dtype=torch.float32)
    ue_v -= (data.dims.ue_v - 1) / 2.0
    ue_v = ue_v.repeat(data.dims.ue_polarizations * data.dims.ue_h)
    candidates = np.arange(-60.0, 60.0 + 0.5, 1.0)
    result = {}
    for name, axis in (("ux", 0), ("uy", 1), ("uz", 2)):
        best = (-1.0, 0.0)
        for candidate in candidates:
            coefficient = torch.exp(
                1j * float(candidate) * delta[:, :, axis, None] * ue_v
            )
            total_cross = torch.einsum("qkn,qkn->", torch.conj(coefficient), cross)
            prediction_energy = torch.einsum(
                "qkn,qkln,qln->", torch.conj(coefficient), gram, coefficient
            ).real
            rho2 = float(
                (torch.abs(total_cross).square() / (prediction_energy * target_energy.sum())).item()
            )
            if rho2 > best[0]:
                best = rho2, float(candidate)
        result[name] = {"coefficient": best[1], "rho_squared": best[0]}
    print(json.dumps(result, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
