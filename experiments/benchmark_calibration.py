from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from physical_ai.data import RoundData
from physical_ai.features import SpectralFeatureLayout
from physical_ai.neighbors import distance_weights, nearest_neighbors


def vector_cos(pred: np.ndarray, target: np.ndarray, axis: int) -> float:
    value = np.sum(pred * target, axis=axis) / np.maximum(
        np.linalg.norm(pred, axis=axis) * np.linalg.norm(target, axis=axis), 1e-30
    )
    return float(value.mean())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate KNN spectral sharpness")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--features", type=Path, default=Path("artifacts/spectral_features.npy"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260804, 101, 202, 303, 404])
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("artifacts/calibration_benchmark.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = RoundData(args.root)
    data.validate()
    positions = np.asarray(data.train_positions)
    features = np.asarray(np.load(args.features, mmap_mode="r"), dtype=np.float32)
    layout = SpectralFeatureLayout.from_dimensions(data.dims)
    gammas = (0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0)
    folds = {}
    for seed in args.seeds:
        permutation = np.random.default_rng(seed).permutation(len(positions))
        val_idx = np.sort(permutation[: args.validation_size])
        train_idx = np.sort(permutation[args.validation_size :])
        local, distances = nearest_neighbors(positions[val_idx], positions[train_idx], 16)
        weights = distance_weights(distances, power=2.0).astype(np.float32)
        prediction = np.einsum("qk,qkd->qd", weights, features[train_idx[local]], optimize=True)
        target = features[val_idx]
        pas_p = prediction[:, : layout.pas_size].reshape(-1, 256, 4)
        pas_t = target[:, : layout.pas_size].reshape(-1, 256, 4)
        pdp_p = prediction[:, layout.pas_size :].reshape(-1, 2, 4, 192)
        pdp_t = target[:, layout.pas_size :].reshape(-1, 2, 4, 192)
        result = {"pas": {}, "pdp": {}}
        for gamma in gammas:
            result["pas"][str(gamma)] = vector_cos(np.maximum(pas_p, 1e-12) ** gamma, pas_t, 1)
            result["pdp"][str(gamma)] = vector_cos(np.maximum(pdp_p, 1e-12) ** gamma, pdp_t, -1)
        folds[str(seed)] = result
    mean = {domain: {} for domain in ("pas", "pdp")}
    for domain in mean:
        for gamma in gammas:
            values = [folds[str(seed)][domain][str(gamma)] for seed in args.seeds]
            mean[domain][str(gamma)] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "folds": values}
    print(json.dumps(mean, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"folds": folds, "mean": mean}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
