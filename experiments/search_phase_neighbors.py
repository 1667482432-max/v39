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
    parser = argparse.ArgumentParser(description="Search radial-phase complex KNN hyperparameters")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/cv_noout_split20260804.pt"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-neighbors", type=int, default=32)
    parser.add_argument("--wavenumber", type=float, default=140.25)
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase_neighbor_search.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = RoundData(".")
    data.validate()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    all_positions = np.asarray(data.train_positions, dtype=np.float64)
    all_features = np.asarray(np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32)
    valid = nonzero_feature_indices(all_features)
    inverse = np.full(len(all_features), -1, dtype=np.int64)
    inverse[valid] = np.arange(len(valid))
    positions = all_positions[valid]
    val_global = np.asarray(checkpoint["validation_indices"], dtype=np.int64)[: args.limit]
    train_global = np.asarray(checkpoint["train_indices"], dtype=np.int64)
    val_idx, train_idx = inverse[val_global], inverse[train_global]
    local, distances = nearest_neighbors(
        positions[val_idx], positions[train_idx], args.max_neighbors
    )
    neighbors = train_idx[local]
    bs = np.array([50.0, 0.0, 25.0])
    radius = np.linalg.norm(positions - bs, axis=1)
    phase = np.exp(
        1j * args.wavenumber * (radius[neighbors] - radius[val_idx, None])
    ).astype(np.complex128)
    q, kmax = neighbors.shape
    cross = np.empty((q, kmax), dtype=np.complex128)
    gram = np.empty((q, kmax, kmax), dtype=np.complex128)
    target_energy = np.empty(q, dtype=np.float64)
    channels = data.train_channels
    device = torch.device("cuda")
    for row in range(q):
        source = torch.from_numpy(
            np.array(channels[valid[neighbors[row]]], dtype=np.complex64, copy=True)
        ).to(device).reshape(kmax, -1)
        target = torch.from_numpy(
            np.array(channels[val_global[row]], dtype=np.complex64, copy=True)
        ).to(device).reshape(-1)
        cross[row] = (torch.conj(source) @ target).cpu().numpy().astype(np.complex128)
        gram[row] = (torch.conj(source) @ source.T).cpu().numpy().astype(np.complex128)
        target_energy[row] = torch.sum(torch.abs(target).square(), dtype=torch.float64).item()
        if (row + 1) % 20 == 0:
            print(f"correlations {row + 1}/{q}", flush=True)
    total_target_energy = target_energy.sum()
    results = {}
    for k in (1, 2, 4, 8, 12, 16, 24, 32):
        if k > kmax:
            continue
        for power in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
            for softening in (0.0, 0.25, 0.5, 1.0, 2.0):
                weight = (distances[:, :k] + softening).clip(1e-6) ** (-power)
                weight /= weight.sum(axis=1, keepdims=True)
                coefficient = weight * phase[:, :k]
                total_cross = np.einsum(
                    "qk,qk->", np.conj(coefficient), cross[:, :k], optimize=True
                )
                prediction_energy = np.einsum(
                    "qk,qkl,ql->", np.conj(coefficient), gram[:, :k, :k], coefficient, optimize=True
                ).real
                rho2 = abs(total_cross) ** 2 / max(prediction_energy * total_target_energy, 1e-30)
                scale = total_cross / max(prediction_energy, 1e-30)
                name = f"k{k}_p{power:g}_e{softening:g}"
                results[name] = {
                    "rho_squared": float(rho2),
                    "nmse": float(1.0 - rho2),
                    "optimal_scale": [float(scale.real), float(scale.imag)],
                }
    top = sorted(results.items(), key=lambda item: item[1]["nmse"])[:50]
    print(json.dumps(top[:20], indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"top": top}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
