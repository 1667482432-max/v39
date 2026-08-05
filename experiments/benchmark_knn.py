from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from physical_ai.data import RoundData
from physical_ai.metrics import StreamingScore
from physical_ai.neighbors import distance_weights, interpolate_complex, nearest_neighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible spatial holdout KNN benchmark")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--powers", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("artifacts/knn_benchmark.json"))
    return parser.parse_args()


def evaluate_candidate(data, val_idx, nn_idx, weights, dims, batch_size: int) -> dict[str, float]:
    score = StreamingScore(dims)
    for start in range(0, len(val_idx), batch_size):
        stop = min(start + batch_size, len(val_idx))
        predictions = np.stack(
            [interpolate_complex(data, nn_idx, weights, i) for i in range(start, stop)]
        )
        targets = np.asarray(data[val_idx[start:stop]], dtype=np.complex64)
        score.update(torch.from_numpy(predictions), torch.from_numpy(targets))
    return score.compute()


def main() -> None:
    args = parse_args()
    round_data = RoundData(args.root)
    round_data.validate()
    positions = np.asarray(round_data.train_positions)
    channels = round_data.train_channels
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(positions))
    val_idx = np.sort(permutation[: args.validation_size])
    train_idx = np.sort(permutation[args.validation_size :])
    max_k = max(args.k)
    neighbors_local, distances = nearest_neighbors(positions[val_idx], positions[train_idx], max_k)
    neighbors_global = train_idx[neighbors_local]
    results: dict[str, dict[str, float]] = {}
    for k in args.k:
        powers = [0.0] if k == 1 else args.powers
        for power in powers:
            if k == 1:
                weights = np.ones((len(val_idx), 1), dtype=np.float64)
            else:
                weights = distance_weights(distances[:, :k], power=power)
            name = f"knn_k{k}_p{power:g}"
            metrics = evaluate_candidate(
                channels, val_idx, neighbors_global[:, :k], weights, round_data.dims, args.batch_size
            )
            results[name] = metrics
            print(name, json.dumps(metrics, sort_keys=True))
    payload = {
        "seed": args.seed,
        "validation_size": len(val_idx),
        "train_size": len(train_idx),
        "nearest_distance_quantiles": np.quantile(
            distances[:, 0], [0, 0.1, 0.5, 0.9, 1]
        ).tolist(),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
