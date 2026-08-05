from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from physical_ai.data import RoundData
from physical_ai.features import SpectralFeatureLayout, nonzero_feature_indices
from physical_ai.neighbors import distance_weights, nearest_neighbors


def cosine_parts(prediction: np.ndarray, target: np.ndarray, layout: SpectralFeatureLayout) -> tuple[float, float]:
    pp = prediction[:, : layout.pas_size].reshape(-1, 256, 4)
    tp = target[:, : layout.pas_size].reshape(-1, 256, 4)
    pd = prediction[:, layout.pas_size :].reshape(-1, 2, 4, 192)
    td = target[:, layout.pas_size :].reshape(-1, 2, 4, 192)
    c1 = np.sum(pp * tp, axis=1) / np.maximum(
        np.linalg.norm(pp, axis=1) * np.linalg.norm(tp, axis=1), 1e-30
    )
    c2 = np.sum(pd * td, axis=-1) / np.maximum(
        np.linalg.norm(pd, axis=-1) * np.linalg.norm(td, axis=-1), 1e-30
    )
    return float(c1.mean()), float(c2.mean())


def graph_prediction(
    labeled_positions: np.ndarray,
    unlabeled_positions: np.ndarray,
    labeled_features: np.ndarray,
    direct_prediction: np.ndarray,
    k: int,
    power: float,
    alpha: float,
) -> np.ndarray:
    labeled_count = len(labeled_positions)
    unlabeled_count = len(unlabeled_positions)
    combined_positions = np.concatenate((labeled_positions, unlabeled_positions), axis=0)
    delta = unlabeled_positions[:, None, :2] - combined_positions[None, :, :2]
    distances = np.linalg.norm(delta, axis=-1)
    distances[np.arange(unlabeled_count), labeled_count + np.arange(unlabeled_count)] = np.inf
    neighbor = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    local_distance = np.take_along_axis(distances, neighbor, axis=1)
    weights = np.maximum(local_distance, 1e-6) ** (-power)
    weights /= weights.sum(axis=1, keepdims=True)
    transition_uu = np.zeros((unlabeled_count, unlabeled_count), dtype=np.float64)
    boundary = np.zeros_like(direct_prediction, dtype=np.float64)
    for row in range(unlabeled_count):
        for index, weight in zip(neighbor[row], weights[row]):
            if index < labeled_count:
                boundary[row] += weight * labeled_features[index]
            else:
                transition_uu[row, index - labeled_count] += weight
    right = (1.0 - alpha) * direct_prediction + alpha * boundary
    return np.linalg.solve(
        np.eye(unlabeled_count, dtype=np.float64) - alpha * transition_uu,
        right,
    ).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transductive graph interpolation on known test coordinates")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--features", type=Path, default=Path("artifacts/spectral_features.npy"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260804, 101, 202, 303, 404])
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("artifacts/transductive_benchmark.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = RoundData(args.root)
    data.validate()
    all_positions = np.asarray(data.train_positions)
    all_features = np.asarray(np.load(args.features, mmap_mode="r"), dtype=np.float32)
    valid = nonzero_feature_indices(all_features)
    positions = all_positions[valid]
    features = all_features[valid]
    layout = SpectralFeatureLayout.from_dimensions(data.dims)
    results = {}
    for seed in args.seeds:
        permutation = np.random.default_rng(seed).permutation(len(positions))
        val_idx = np.sort(permutation[: args.validation_size])
        train_idx = np.sort(permutation[args.validation_size :])
        direct_local, direct_distance = nearest_neighbors(
            positions[val_idx], positions[train_idx], 16
        )
        direct_weights = distance_weights(direct_distance, power=2.0).astype(np.float32)
        direct = np.einsum(
            "qk,qkd->qd", direct_weights, features[train_idx[direct_local]], optimize=True
        )
        fold = {}
        c1, c2 = cosine_parts(direct, features[val_idx], layout)
        fold["direct"] = {"pas": c1, "pdp": c2, "mean": 0.5 * (c1 + c2)}
        for k in (8, 16, 24, 32):
            for alpha in (0.25, 0.5, 0.75, 1.0):
                prediction = graph_prediction(
                    positions[train_idx],
                    positions[val_idx],
                    features[train_idx],
                    direct,
                    k=k,
                    power=2.0,
                    alpha=alpha,
                )
                c1, c2 = cosine_parts(prediction, features[val_idx], layout)
                fold[f"k{k}_a{alpha:g}"] = {
                    "pas": c1,
                    "pdp": c2,
                    "mean": 0.5 * (c1 + c2),
                }
        results[str(seed)] = fold
        best = max(fold.items(), key=lambda item: item[1]["mean"])
        print(seed, best[0], json.dumps(best[1]), flush=True)
    names = results[str(args.seeds[0])].keys()
    summary = {}
    for name in names:
        summary[name] = {
            metric: float(np.mean([results[str(seed)][name][metric] for seed in args.seeds]))
            for metric in ("pas", "pdp", "mean")
        }
    print("TOP MEAN")
    for name, values in sorted(summary.items(), key=lambda item: item[1]["mean"], reverse=True):
        print(name, json.dumps(values))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"folds": results, "summary": summary}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
