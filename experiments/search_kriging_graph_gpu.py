from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from physical_ai.features import SpectralFeatureLayout, nonzero_feature_indices
from experiments.search_improved_graph_gpu import graph_matrices
from experiments.search_kriging_ensemble_gpu import Config, predict_config
from experiments.search_spatial_kernels_gpu import cosine_parts, metric_embeddings


PAS_CONFIG = Config("xy_ctx-patch_s4", "exponential", 16, 0.5, 0.01, True)
PDP_CONFIG = Config("xy_ctx-patch_s4", "exponential", 24, 0.75, 0.05, False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph propagation on kriging predictions")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260804, 101, 202, 303, 404])
    parser.add_argument("--validation-size", type=int, default=400)
    parser.add_argument("--output", type=Path, default=Path("artifacts/kriging_graph_gpu.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    all_positions = np.load("Round1_Train_Pos.npy").astype(np.float32)
    all_contexts = np.load("artifacts/map_context.npz")["train"].astype(np.float32)
    all_features = np.asarray(np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32)
    valid = nonzero_feature_indices(all_features)
    positions = all_positions[valid]
    contexts = all_contexts[valid]
    features_np = all_features[valid]
    features = torch.from_numpy(features_np).to(device)
    embeddings = metric_embeddings(positions, contexts)
    selected = {
        name: embeddings[name]
        for name in (
            "xy_y1", "xy_y0.75", "xy_ctx-summary_s2", "xy_ctx-summary_s4",
            "xy_ctx-patch_s2", "xy_ctx-patch_s4", "xy_ctx-all_s2", "xy_ctx-all_s4",
        )
    }
    layout = SpectralFeatureLayout(1024, 1536)
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    fold_bests = {}
    for seed in args.seeds:
        permutation = np.random.default_rng(seed).permutation(len(features_np))
        query_idx = np.sort(permutation[: args.validation_size])
        train_idx = np.sort(permutation[args.validation_size :])
        pas_prediction = predict_config(
            PAS_CONFIG, embeddings[PAS_CONFIG.metric], train_idx, query_idx, features
        )
        pdp_prediction = predict_config(
            PDP_CONFIG, embeddings[PDP_CONFIG.metric], train_idx, query_idx, features
        )
        direct = torch.cat((pas_prediction[:, :1024], pdp_prediction[:, 1024:]), dim=1)
        target = features[query_idx]
        pas, pdp = cosine_parts(direct, target, layout)
        direct_mean = 0.5 * (pas + pdp)
        fold_results = {"direct_kriging": {"pas": pas, "pdp": pdp, "mean": direct_mean}}
        totals["direct_kriging"][0] += pas
        totals["direct_kriging"][1] += pdp
        totals["direct_kriging"][2] += direct_mean
        for metric_name, embedding in selected.items():
            for k in (8, 12, 16, 24):
                for power in (1.5, 2.0, 2.5):
                    transition, boundary = graph_matrices(
                        embedding, train_idx, query_idx, features, k, power, softening=0.0
                    )
                    identity = torch.eye(len(query_idx), device=device)
                    for alpha in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
                        right = (1.0 - alpha) * direct + alpha * boundary
                        prediction = torch.linalg.solve(identity - alpha * transition, right)
                        pas, pdp = cosine_parts(prediction, target, layout)
                        name = f"{metric_name}__k{k}_p{power:g}_a{alpha:g}"
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
        metric: sorted(summary.items(), key=lambda item: item[1][metric], reverse=True)[:50]
        for metric in ("pas", "pdp", "mean")
    }
    print("TOP", json.dumps({key: value[:10] for key, value in top.items()}, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"fold_bests": fold_bests, "top": top}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
