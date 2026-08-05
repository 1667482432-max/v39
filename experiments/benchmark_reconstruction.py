from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from physical_ai.data import RoundData
from physical_ai.metrics import StreamingScore
from physical_ai.neighbors import distance_weights, nearest_neighbors
from physical_ai.spectral import (
    alternating_spectral_projection,
    knn_spectral_targets,
    physical_axis_denoise,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PAS/PDP alternating reconstruction")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--validation-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Only evaluate one projection per final axis")
    parser.add_argument("--output", type=Path, default=Path("artifacts/reconstruction_benchmark.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = RoundData(args.root)
    data.validate()
    positions = np.asarray(data.train_positions)
    channels = data.train_channels
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(positions))
    val_idx, train_idx = np.sort(order[: args.validation_size]), np.sort(order[args.validation_size :])
    nn_local, distances = nearest_neighbors(positions[val_idx], positions[train_idx], args.k)
    nn_global = train_idx[nn_local]
    weights = distance_weights(distances, power=args.power)
    pdp_weights = distance_weights(distances[:, :4], power=1.0)
    configs = [(0, 1.0, "complex")]
    if args.quick:
        configs.extend([(1, 1.0, "pas"), (1, 1.0, "pdp")])
    else:
        for relaxation in (0.5, 1.0):
            for iterations in (1, 2, 5, 10, 20):
                for final in ("pas", "pdp"):
                    configs.append((iterations, relaxation, final))
    scores = {config: StreamingScore(data.dims) for config in configs}
    for qi, target_idx in enumerate(val_idx):
        neighbor_h = torch.from_numpy(
            np.asarray(channels[nn_global[qi]], dtype=np.complex64)
        )
        local_w = torch.from_numpy(weights[qi].astype(np.float32))
        view = local_w.view(-1, 1, 1, 1)
        initial = torch.sum(neighbor_h * view, dim=0)
        target_pas, _ = knn_spectral_targets(neighbor_h, local_w)
        _, target_pdp = knn_spectral_targets(
            neighbor_h[:4], torch.from_numpy(pdp_weights[qi].astype(np.float32))
        )
        if args.denoise:
            target_pas, target_pdp = physical_axis_denoise(target_pas, target_pdp)
        target = torch.from_numpy(np.asarray(channels[target_idx], dtype=np.complex64)).unsqueeze(0)
        for config in configs:
            iterations, relaxation, final = config
            if final == "complex":
                prediction = initial
            else:
                prediction = alternating_spectral_projection(
                    initial, target_pas, target_pdp, iterations, relaxation, final
                )
            scores[config].update(prediction.unsqueeze(0), target)
        if (qi + 1) % 10 == 0:
            print(f"processed {qi + 1}/{len(val_idx)}", flush=True)
    results = {}
    for config, accumulator in scores.items():
        iterations, relaxation, final = config
        name = "complex_knn" if final == "complex" else f"ap_{final}_i{iterations}_r{relaxation:g}"
        results[name] = accumulator.compute()
    for name, metrics in sorted(
        results.items(), key=lambda item: item[1]["score_at_optimal_scale"], reverse=True
    ):
        print(name, json.dumps(metrics, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"validation_size": len(val_idx), "results": results}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
