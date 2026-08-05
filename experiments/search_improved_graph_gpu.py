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


PAS_METRIC = "xy_ctx-all_s4"
PDP_METRIC = "xy_ctx-patch_s4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU graph search on improved map-aware predictions")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260804, 101, 202])
    parser.add_argument("--validation-size", type=int, default=400)
    parser.add_argument("--output", type=Path, default=Path("artifacts/improved_graph_gpu_search.json"))
    return parser.parse_args()


def direct_prediction(
    embedding: np.ndarray,
    train_idx: np.ndarray,
    query_idx: np.ndarray,
    features: torch.Tensor,
    k: int = 16,
    power: float = 3.0,
    softening: float = 1.0,
) -> torch.Tensor:
    distance, local = cKDTree(embedding[train_idx]).query(embedding[query_idx], k=k, workers=-1)
    weight = torch.from_numpy((distance + softening).astype(np.float32)).to(features.device).pow(-power)
    weight /= weight.sum(dim=1, keepdim=True)
    neighbor = torch.from_numpy(train_idx[local].astype(np.int64)).to(features.device)
    return torch.einsum("qk,qkd->qd", weight, features[neighbor])


def graph_matrices(
    embedding: np.ndarray,
    train_idx: np.ndarray,
    query_idx: np.ndarray,
    features: torch.Tensor,
    k: int,
    power: float,
    softening: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined_idx = np.concatenate((train_idx, query_idx))
    combined_embedding = embedding[combined_idx]
    query_embedding = embedding[query_idx]
    distances, neighbors = cKDTree(combined_embedding).query(query_embedding, k=k + 1, workers=-1)
    # The first match is the query row itself.
    distances = distances[:, 1:]
    neighbors = neighbors[:, 1:]
    weights = (distances + softening) ** (-power)
    weights /= weights.sum(axis=1, keepdims=True)
    q = len(query_idx)
    n_labeled = len(train_idx)
    transition = np.zeros((q, q), dtype=np.float32)
    boundary = torch.zeros((q, features.shape[1]), dtype=torch.float32, device=features.device)
    for row in range(q):
        labeled_mask = neighbors[row] < n_labeled
        if np.any(labeled_mask):
            local_weights = torch.from_numpy(weights[row, labeled_mask].astype(np.float32)).to(features.device)
            labeled_global = torch.from_numpy(train_idx[neighbors[row, labeled_mask]].astype(np.int64)).to(
                features.device
            )
            boundary[row] = torch.einsum("k,kd->d", local_weights, features[labeled_global])
        unlabeled = neighbors[row, ~labeled_mask] - n_labeled
        transition[row, unlabeled] += weights[row, ~labeled_mask].astype(np.float32)
    return torch.from_numpy(transition).to(features.device), boundary


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
    embeddings = metric_embeddings(positions, contexts)
    selected_metrics = {
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
        direct_pas = direct_prediction(embeddings[PAS_METRIC], train_idx, query_idx, features)
        direct_pdp = direct_prediction(embeddings[PDP_METRIC], train_idx, query_idx, features)
        direct = torch.cat((direct_pas[:, : layout.pas_size], direct_pdp[:, layout.pas_size :]), dim=1)
        target = features[query_idx]
        pas, pdp = cosine_parts(direct, target, layout)
        fold_results = {"direct_improved": {"pas": pas, "pdp": pdp, "mean": 0.5 * (pas + pdp)}}
        totals["direct_improved"][0] += pas
        totals["direct_improved"][1] += pdp
        totals["direct_improved"][2] += 0.5 * (pas + pdp)
        for metric_name, embedding in selected_metrics.items():
            for k in (8, 12, 16, 24):
                for power in (1.5, 2.0, 2.5, 3.0):
                    transition, boundary = graph_matrices(
                        embedding, train_idx, query_idx, features, k, power, softening=0.0
                    )
                    identity = torch.eye(len(query_idx), device=device)
                    for alpha in (0.1, 0.25, 0.4, 0.5, 0.75):
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
        name: {"pas": values[0] / count, "pdp": values[1] / count, "mean": values[2] / count}
        for name, values in totals.items()
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
