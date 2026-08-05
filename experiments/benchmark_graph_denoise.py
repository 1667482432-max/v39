from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from physical_ai.data import RoundData
from physical_ai.features import SpectralFeatureLayout
from physical_ai.neighbors import distance_weights, nearest_neighbors


def cosine_parts(prediction: np.ndarray, target: np.ndarray, layout: SpectralFeatureLayout) -> tuple[float, float]:
    pp = prediction[:, : layout.pas_size].reshape(-1, 256, 4)
    tp = target[:, : layout.pas_size].reshape(-1, 256, 4)
    pd = prediction[:, layout.pas_size :].reshape(-1, 2, 4, 192)
    td = target[:, layout.pas_size :].reshape(-1, 2, 4, 192)
    c1 = np.sum(pp * tp, axis=1) / np.maximum(np.linalg.norm(pp, axis=1) * np.linalg.norm(tp, axis=1), 1e-30)
    c2 = np.sum(pd * td, axis=-1) / np.maximum(np.linalg.norm(pd, axis=-1) * np.linalg.norm(td, axis=-1), 1e-30)
    return float(c1.mean()), float(c2.mean())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph diffusion denoising benchmark")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--features", type=Path, default=Path("artifacts/spectral_features.npy"))
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output", type=Path, default=Path("artifacts/graph_benchmark.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = RoundData(args.root)
    data.validate()
    positions = np.asarray(data.train_positions)
    features = np.asarray(np.load(args.features, mmap_mode="r"), dtype=np.float32)
    layout = SpectralFeatureLayout.from_dimensions(data.dims)
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(positions))
    val_idx, train_idx = np.sort(permutation[: args.validation_size]), np.sort(permutation[args.validation_size :])
    train_local, train_dist = nearest_neighbors(positions[train_idx], positions[train_idx], 33)
    train_local, train_dist = train_local[:, 1:], train_dist[:, 1:]
    query_local, query_dist = nearest_neighbors(positions[val_idx], positions[train_idx], 16)
    query_weights = distance_weights(query_dist, power=2.0).astype(np.float32)
    target = features[val_idx]
    raw_bank = features[train_idx]
    results = {}
    for k in (4, 8, 16, 24, 32):
        graph_weights = distance_weights(train_dist[:, :k], power=2.0).astype(np.float32)
        graph_prediction = np.einsum(
            "qk,qkd->qd", graph_weights, raw_bank[train_local[:, :k]], optimize=True
        )
        for blend in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
            bank = raw_bank * (1.0 - blend) + graph_prediction * blend
            prediction = np.einsum(
                "qk,qkd->qd", query_weights, bank[query_local], optimize=True
            )
            c1, c2 = cosine_parts(prediction, target, layout)
            results[f"k{k}_b{blend:g}"] = {
                "pas": c1,
                "pdp": c2,
                "mean": 0.5 * (c1 + c2),
            }
    for name, metrics in sorted(results.items(), key=lambda item: item[1]["mean"], reverse=True):
        print(name, json.dumps(metrics))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
