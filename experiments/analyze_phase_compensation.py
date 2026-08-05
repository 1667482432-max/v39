from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from physical_ai.data import RoundData
from physical_ai.features import nonzero_feature_indices
from physical_ai.neighbors import distance_weights, nearest_neighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search physics-informed radial carrier phase compensation")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/cv_noout_split20260804.pt"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--max-wavenumber", type=float, default=400.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase_compensation.json"))
    return parser.parse_args()


def statistics(
    coefficient: np.ndarray,
    cross: np.ndarray,
    gram: np.ndarray,
    target_energy: np.ndarray,
    rows: np.ndarray,
) -> tuple[complex, float, float]:
    local = coefficient[:, rows]
    total_cross = np.einsum("kq,qk->", np.conj(local), cross[rows], optimize=True)
    prediction_energy = np.einsum(
        "kq,qkl,lq->", np.conj(local), gram[rows], local, optimize=True
    ).real
    energy = float(target_energy[rows].sum())
    return total_cross, float(prediction_energy), energy


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
    local, distance = nearest_neighbors(positions[val_idx], positions[train_idx], 16)
    neighbors = train_idx[local]
    weights = distance_weights(distance, power=2.0).astype(np.float64)
    bs = np.array([50.0, 0.0, 25.0])
    radius = np.linalg.norm(positions - bs, axis=1)
    radial_delta = radius[neighbors] - radius[val_idx, None]
    cross = np.empty((len(val_idx), 16), dtype=np.complex128)
    gram = np.empty((len(val_idx), 16, 16), dtype=np.complex128)
    target_energy = np.empty(len(val_idx), dtype=np.float64)
    channels = data.train_channels
    device = torch.device("cuda")
    for row in range(len(val_idx)):
        source = torch.from_numpy(
            np.array(channels[valid[neighbors[row]]], dtype=np.complex64, copy=True)
        ).to(device).reshape(16, -1)
        target = torch.from_numpy(
            np.array(channels[val_global[row]], dtype=np.complex64, copy=True)
        ).to(device).reshape(-1)
        cross[row] = (torch.conj(source) @ target).cpu().numpy().astype(np.complex128)
        gram[row] = (torch.conj(source) @ source.T).cpu().numpy().astype(np.complex128)
        target_energy[row] = torch.sum(torch.abs(target).square(), dtype=torch.float64).item()
        if (row + 1) % 20 == 0:
            print(f"correlations {row + 1}/{len(val_idx)}", flush=True)
    wavenumbers = np.arange(
        -args.max_wavenumber, args.max_wavenumber + args.step * 0.5, args.step, dtype=np.float64
    )
    coefficient = weights.T[:, :, None] * np.exp(
        1j * wavenumbers[None, None, :] * radial_delta.T[:, :, None]
    )
    # Reorder to (K, Q, W) and evaluate each wavenumber.
    all_rows = np.arange(len(val_idx))
    rho2 = np.empty(len(wavenumbers), dtype=np.float64)
    for index in range(len(wavenumbers)):
        c, ep, et = statistics(coefficient[:, :, index], cross, gram, target_energy, all_rows)
        rho2[index] = abs(c) ** 2 / max(ep * et, 1e-30)
    best_index = int(np.argmax(rho2))
    rng = np.random.default_rng(20260804)
    permutation = rng.permutation(len(val_idx))
    splits = np.array_split(permutation, 5)
    heldout = []
    for fold in range(5):
        test_rows = splits[fold]
        train_rows = np.concatenate([splits[i] for i in range(5) if i != fold])
        train_quality = []
        train_stats = []
        for index in range(len(wavenumbers)):
            stat = statistics(coefficient[:, :, index], cross, gram, target_energy, train_rows)
            train_stats.append(stat)
            train_quality.append(abs(stat[0]) ** 2 / max(stat[1] * stat[2], 1e-30))
        selected = int(np.argmax(train_quality))
        train_cross, train_prediction_energy, _ = train_stats[selected]
        scale = train_cross / max(train_prediction_energy, 1e-30)
        test_cross, test_prediction_energy, test_energy = statistics(
            coefficient[:, :, selected], cross, gram, target_energy, test_rows
        )
        nmse = (
            test_energy + abs(scale) ** 2 * test_prediction_energy
            - 2.0 * np.real(np.conj(scale) * test_cross)
        ) / test_energy
        heldout.append(
            {"fold": fold, "wavenumber": float(wavenumbers[selected]), "scale": [scale.real, scale.imag], "nmse": float(nmse)}
        )
    result = {
        "global_best_wavenumber": float(wavenumbers[best_index]),
        "global_rho_squared": float(rho2[best_index]),
        "global_complex_optimal_nmse": float(1.0 - rho2[best_index]),
        "zero_wavenumber_rho_squared": float(rho2[np.argmin(np.abs(wavenumbers))]),
        "heldout": heldout,
        "heldout_mean_nmse": float(np.mean([item["nmse"] for item in heldout])),
    }
    print(json.dumps(result, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
