from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from physical_ai.features import SpectralFeatureLayout, nonzero_feature_indices
from experiments.search_spatial_kernels_gpu import cosine_parts, metric_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU local ordinary-kriging search")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260804, 101, 202])
    parser.add_argument("--validation-size", type=int, default=400)
    parser.add_argument("--output", type=Path, default=Path("artifacts/local_kriging_gpu_search.json"))
    return parser.parse_args()


def covariance(distance: torch.Tensor, bandwidth: torch.Tensor, kind: str) -> torch.Tensor:
    ratio = distance / bandwidth.clamp_min(1e-5)
    if kind == "gaussian":
        return torch.exp(-0.5 * ratio.square())
    if kind == "exponential":
        return torch.exp(-ratio)
    root3 = 3.0 ** 0.5
    return (1.0 + root3 * ratio) * torch.exp(-root3 * ratio)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    all_positions = np.load("Round1_Train_Pos.npy").astype(np.float32)
    all_contexts = np.load("artifacts/map_context.npz")["train"].astype(np.float32)
    all_features = np.asarray(np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32)
    valid = nonzero_feature_indices(all_features)
    positions = all_positions[valid]
    contexts = all_contexts[valid]
    features_np = all_features[valid]
    features = torch.from_numpy(features_np).to(device)
    all_embeddings = metric_embeddings(positions, contexts)
    embeddings = {
        name: all_embeddings[name]
        for name in (
            "xy_y1", "xy_y0.75", "xy_ctx-summary_s4", "xy_ctx-patch_s4", "xy_ctx-all_s4"
        )
    }
    layout = SpectralFeatureLayout(1024, 1536)
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    fold_bests = {}
    for seed in args.seeds:
        permutation = np.random.default_rng(seed).permutation(len(features_np))
        query_idx = np.sort(permutation[: args.validation_size])
        train_idx = np.sort(permutation[args.validation_size :])
        target = features[query_idx]
        fold_results = {}
        for metric_name, embedding in embeddings.items():
            distance_np, local = cKDTree(embedding[train_idx]).query(
                embedding[query_idx], k=32, workers=-1
            )
            neighbor_all = train_idx[local]
            for k in (8, 12, 16, 24, 32):
                neighbor_np = neighbor_all[:, :k]
                neighbor_embedding = embedding[neighbor_np]
                pair_np = np.linalg.norm(
                    neighbor_embedding[:, :, None, :] - neighbor_embedding[:, None, :, :], axis=-1
                ).astype(np.float32)
                pair = torch.from_numpy(pair_np).to(device)
                query_distance = torch.from_numpy(distance_np[:, :k].astype(np.float32)).to(device)
                source = features[torch.from_numpy(neighbor_np.astype(np.int64)).to(device)]
                ones = torch.ones((len(query_idx), k), device=device)
                for kind in ("gaussian", "exponential", "matern32"):
                    for scale in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
                        bandwidth = query_distance[:, -1:, None] * scale
                        cov_nn = covariance(pair, bandwidth, kind)
                        cov_q = covariance(query_distance, bandwidth[:, :, 0], kind)
                        for nugget in (1e-3, 1e-2, 5e-2, 0.1, 0.25, 0.5):
                            system = torch.zeros((len(query_idx), k + 1, k + 1), device=device)
                            system[:, :k, :k] = cov_nn
                            system[:, :k, :k] += torch.eye(k, device=device) * nugget
                            system[:, :k, k] = 1.0
                            system[:, k, :k] = 1.0
                            right = torch.cat((cov_q, torch.ones((len(query_idx), 1), device=device)), dim=1)
                            weight = torch.linalg.solve(system, right)[..., :k]
                            for mode in ("raw", "positive"):
                                local_weight = weight
                                if mode == "positive":
                                    local_weight = weight.clamp_min(0.0)
                                    local_weight /= local_weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
                                prediction = torch.einsum("qk,qkd->qd", local_weight, source)
                                prediction = prediction.clamp_min(0.0)
                                pas, pdp = cosine_parts(prediction, target, layout)
                                name = (
                                    f"{metric_name}__{kind}_k{k}_s{scale:g}_n{nugget:g}_{mode}"
                                )
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
