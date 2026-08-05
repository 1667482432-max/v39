from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from physical_ai.data import RoundData
from physical_ai.features import spectral_targets_from_features
from physical_ai.neighbors import distance_weights, nearest_neighbors
from physical_ai.spectral import alternating_spectral_projection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune path-loss restoration without changing spectra")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--features", type=Path, default=Path("artifacts/spectral_features.npy"))
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output", type=Path, default=Path("artifacts/energy_benchmark.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = RoundData(args.root)
    data.validate()
    positions = np.asarray(data.train_positions)
    features = np.asarray(np.load(args.features, mmap_mode="r"), dtype=np.float32)
    channels = data.train_channels
    permutation = np.random.default_rng(args.seed).permutation(len(positions))
    val_idx = np.sort(permutation[: args.validation_size])
    train_idx = np.sort(permutation[args.validation_size :])
    local, distances = nearest_neighbors(positions[val_idx], positions[train_idx], 16)
    neighbor_idx = train_idx[local]
    weights = distance_weights(distances, power=2.0).astype(np.float32)
    predicted_features = np.einsum(
        "qk,qkd->qd", weights, features[neighbor_idx], optimize=True
    )
    totals = defaultdict(lambda: {"p": 0.0, "c": 0.0, "t": 0.0, "energy_log_error": 0.0})
    for i, target_idx in enumerate(val_idx):
        source = torch.from_numpy(np.array(channels[neighbor_idx[i]], dtype=np.complex64, copy=True))
        local_w = torch.from_numpy(weights[i])
        initial = torch.sum(source * local_w.view(-1, 1, 1, 1), dim=0)
        pas, pdp = spectral_targets_from_features(
            torch.from_numpy(predicted_features[i]), data.dims
        )
        reconstructed = alternating_spectral_projection(
            initial, pas, pdp, iterations=20, relaxation=0.5, final_constraint="pdp"
        )
        target = torch.from_numpy(np.array(channels[target_idx], dtype=np.complex64, copy=True))
        source_energy = torch.sum(torch.abs(source).square(), dim=(1, 2, 3)).double()
        target_energy = torch.sum(torch.abs(target).square(), dtype=torch.float64).item()
        raw_energy = torch.sum(torch.abs(reconstructed).square(), dtype=torch.float64).item()
        estimates = {
            "constant": raw_energy,
            "nearest": source_energy[0].item(),
            "arithmetic": torch.sum(source_energy * local_w.double()).item(),
            "geometric": torch.exp(torch.sum(torch.log(source_energy) * local_w.double())).item(),
        }
        for name, estimate in estimates.items():
            scaled = reconstructed * np.sqrt(estimate / max(raw_energy, 1e-30))
            totals[name]["p"] += torch.sum(torch.abs(scaled).square(), dtype=torch.float64).item()
            totals[name]["c"] += torch.sum(
                torch.real(torch.conj(scaled) * target), dtype=torch.float64
            ).item()
            totals[name]["t"] += target_energy
            totals[name]["energy_log_error"] += (
                np.log(max(estimate, 1e-30)) - np.log(max(target_energy, 1e-30))
            ) ** 2
        if (i + 1) % 20 == 0:
            print(f"processed {i + 1}/{len(val_idx)}", flush=True)
    results = {}
    for name, value in totals.items():
        rho = max(0.0, value["c"] / max(value["p"], 1e-30))
        nmse = (value["t"] + rho * rho * value["p"] - 2 * rho * value["c"]) / value["t"]
        results[name] = {
            "optimal_shrinkage": rho,
            "optimal_nmse": nmse,
            "nmse_score_term": 0.2 / (1.0 + nmse),
            "energy_log_rmse": np.sqrt(value["energy_log_error"] / len(val_idx)),
        }
    print(json.dumps(results, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
