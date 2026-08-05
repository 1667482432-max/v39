from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from physical_ai.features import SpectralFeatureLayout, nonzero_feature_indices
from experiments.search_kriging_ensemble_gpu import Config, predict_config
from experiments.search_spatial_kernels_gpu import cosine_parts


KERNELS = (
    Config("custom", "exponential", 16, 0.5, 0.01, True),
    Config("custom", "exponential", 32, 0.5, 0.05, False),
    Config("custom", "exponential", 24, 0.75, 0.05, False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search separate map-height/density metric scales")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260804])
    parser.add_argument("--validation-size", type=int, default=400)
    parser.add_argument("--refine", action="store_true", help="Evaluate the focused top metric region")
    parser.add_argument("--output", type=Path, default=Path("artifacts/map_metric_kriging_gpu.json"))
    return parser.parse_args()


def standardized(raw: np.ndarray) -> np.ndarray:
    return (raw - raw.mean(axis=0, keepdims=True)) / np.maximum(raw.std(axis=0, keepdims=True), 1e-3)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    all_positions = np.load("Round1_Train_Pos.npy").astype(np.float32)
    all_contexts = np.load("artifacts/map_context.npz")["train"].astype(np.float32)
    all_features = np.asarray(np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32)
    valid = nonzero_feature_indices(all_features)
    xy = all_positions[valid, :2].astype(np.float64)
    contexts = all_contexts[valid]
    patch_height = standardized(contexts[:, 103:128]) / np.sqrt(25.0)
    patch_density = standardized(contexts[:, 128:153]) / np.sqrt(25.0)
    features_np = all_features[valid]
    features = torch.from_numpy(features_np).to(device)
    if args.refine:
        y_scales = (0.75, 1.0, 1.25)
        height_scales = (0.0, 1.0, 2.0, 2.5, 3.0)
        density_scales = (2.5, 3.0, 3.5, 4.0)
    else:
        y_scales = (0.5, 0.75, 1.0, 1.25, 1.5)
        height_scales = density_scales = (0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)
    metrics = {}
    for y_scale in y_scales:
        scaled_xy = xy * np.array([1.0, y_scale])
        for height_scale in height_scales:
            for density_scale in density_scales:
                name = f"y{y_scale:g}_h{height_scale:g}_d{density_scale:g}"
                metrics[name] = np.concatenate(
                    (scaled_xy, patch_height * height_scale, patch_density * density_scale), axis=1
                )
    layout = SpectralFeatureLayout(1024, 1536)
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    fold_bests = {}
    for seed in args.seeds:
        permutation = np.random.default_rng(seed).permutation(len(features_np))
        query_idx = np.sort(permutation[: args.validation_size])
        train_idx = np.sort(permutation[args.validation_size :])
        target = features[query_idx]
        fold_results = {}
        for metric_name, embedding in metrics.items():
            for kernel in KERNELS:
                prediction = predict_config(kernel, embedding, train_idx, query_idx, features)
                pas, pdp = cosine_parts(prediction, target, layout)
                name = f"{metric_name}__{kernel.name}"
                mean = 0.5 * (pas + pdp)
                fold_results[name] = {"pas": pas, "pdp": pdp, "mean": mean}
                totals[name][0] += pas
                totals[name][1] += pdp
                totals[name][2] += mean
        fold_bests[str(seed)] = {
            metric: max(fold_results.items(), key=lambda item: item[1][metric])
            for metric in ("pas", "pdp", "mean")
        }
        print(seed, json.dumps(fold_bests[str(seed)]), flush=True)
    count = len(args.seeds)
    summary = {
        name: {"pas": value[0] / count, "pdp": value[1] / count, "mean": value[2] / count}
        for name, value in totals.items()
    }
    top = {
        metric: sorted(summary.items(), key=lambda item: item[1][metric], reverse=True)[:100]
        for metric in ("pas", "pdp", "mean")
    }
    print("TOP", json.dumps({key: value[:10] for key, value in top.items()}, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"fold_bests": fold_bests, "top": top}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
