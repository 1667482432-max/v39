from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from physical_ai.features import SpectralFeatureLayout, nonzero_feature_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Large GPU search of Physical-AI spatial kernels")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260804, 101, 202])
    parser.add_argument("--validation-size", type=int, default=400)
    parser.add_argument("--max-neighbors", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("artifacts/spatial_kernel_gpu_search.json"))
    return parser.parse_args()


def metric_embeddings(positions: np.ndarray, contexts: np.ndarray) -> dict[str, np.ndarray]:
    xy = positions[:, :2].astype(np.float64)
    relative = xy - np.array([50.0, 0.0])
    radius = np.linalg.norm(relative, axis=1)
    angle = np.unwrap(np.arctan2(relative[:, 1], relative[:, 0]))
    embeddings: dict[str, np.ndarray] = {}
    for y_scale in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        embeddings[f"xy_y{y_scale:g}"] = xy * np.array([1.0, y_scale])
    # Locally Euclidean radial/tangential coordinates with tunable angular scale.
    median_radius = np.median(radius)
    for angular_scale in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        embeddings[f"polar_a{angular_scale:g}"] = np.column_stack(
            (radius, angle * median_radius * angular_scale)
        )
    context_groups = {
        "summary": contexts[:, :7],
        "corridor": contexts[:, 7:103],
        "clearance": contexts[:, 71:103],
        "patch": contexts[:, 103:153],
        "all": contexts,
    }
    for group_name, raw in context_groups.items():
        centered = raw - np.mean(raw, axis=0, keepdims=True)
        scaled = centered / np.maximum(np.std(centered, axis=0, keepdims=True), 1e-3)
        scaled /= np.sqrt(scaled.shape[1])
        for context_scale in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
            embeddings[f"xy_ctx-{group_name}_s{context_scale:g}"] = np.concatenate(
                (xy, scaled * context_scale), axis=1
            )
    return embeddings


def cosine_parts(
    prediction: torch.Tensor,
    target: torch.Tensor,
    layout: SpectralFeatureLayout,
) -> tuple[float, float]:
    pas_p = prediction[:, : layout.pas_size].reshape(-1, 256, 4)
    pas_t = target[:, : layout.pas_size].reshape(-1, 256, 4)
    pdp_p = prediction[:, layout.pas_size :].reshape(-1, 2, 4, 192)
    pdp_t = target[:, layout.pas_size :].reshape(-1, 2, 4, 192)
    pas = torch.nn.functional.cosine_similarity(pas_p, pas_t, dim=1).mean().item()
    pdp = torch.nn.functional.cosine_similarity(pdp_p, pdp_t, dim=-1).mean().item()
    return pas, pdp


def candidate_weights(distances: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    result: list[tuple[str, torch.Tensor]] = []
    for k in (4, 8, 12, 16, 24, 32, 48, 64):
        local = distances[:, :k]
        for power in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
            for softening in (0.0, 0.25, 0.5, 1.0):
                weight = (local + softening).clamp_min(1e-6).pow(-power)
                weight /= weight.sum(dim=1, keepdim=True)
                result.append((f"idw_k{k}_p{power:g}_e{softening:g}", weight))
        bandwidth = local[:, -1:].clamp_min(1e-6)
        for scale in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
            gaussian = torch.exp(-0.5 * (local / (bandwidth * scale)).square())
            gaussian /= gaussian.sum(dim=1, keepdim=True)
            result.append((f"gauss_k{k}_s{scale:g}", gaussian))
            exponential = torch.exp(-local / (bandwidth * scale))
            exponential /= exponential.sum(dim=1, keepdim=True)
            result.append((f"exp_k{k}_s{scale:g}", exponential))
    return result


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
    features = all_features[valid]
    embeddings = metric_embeddings(positions, contexts)
    layout = SpectralFeatureLayout(pas_size=1024, pdp_size=1536)
    feature_t = torch.from_numpy(features).to(device)
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    fold_bests = {}
    for seed in args.seeds:
        permutation = np.random.default_rng(seed).permutation(len(features))
        val_idx = np.sort(permutation[: args.validation_size])
        train_idx = np.sort(permutation[args.validation_size :])
        target = feature_t[val_idx]
        fold_results: dict[str, dict[str, float]] = {}
        for metric_name, embedding in embeddings.items():
            tree = cKDTree(embedding[train_idx])
            distances_np, neighbor_local = tree.query(
                embedding[val_idx], k=args.max_neighbors, workers=-1
            )
            neighbor = torch.from_numpy(train_idx[neighbor_local].astype(np.int64)).to(device)
            distances = torch.from_numpy(distances_np.astype(np.float32)).to(device)
            source = feature_t[neighbor]
            for weight_name, weight in candidate_weights(distances):
                k = weight.shape[1]
                prediction = torch.einsum("qk,qkd->qd", weight, source[:, :k])
                pas, pdp = cosine_parts(prediction, target, layout)
                name = f"{metric_name}__{weight_name}"
                mean = 0.5 * (pas + pdp)
                fold_results[name] = {"pas": pas, "pdp": pdp, "mean": mean}
                totals[name][0] += pas
                totals[name][1] += pdp
                totals[name][2] += mean
            del source, neighbor, distances
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
    args.output.write_text(
        json.dumps({"fold_bests": fold_bests, "top": top}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
