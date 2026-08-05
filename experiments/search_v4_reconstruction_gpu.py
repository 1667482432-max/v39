from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from physical_ai.data import RoundData
from physical_ai.features import nonzero_feature_indices, spectral_targets_from_features
from physical_ai.metrics import score_components
from physical_ai.neighbors import distance_weights, nearest_neighbors
from experiments.search_improved_graph_gpu import graph_matrices
from experiments.search_kriging_ensemble_gpu import Config, predict_config
from experiments.search_spatial_kernels_gpu import metric_embeddings


PAS_CONFIG = Config("xy_ctx-patch_s4", "exponential", 16, 0.5, 0.01, True)
PDP_CONFIG = Config("xy_ctx-patch_s4", "exponential", 24, 0.75, 0.05, False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V4 reconstruction hyperparameter search")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/cv_noout_split20260804.pt"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("artifacts/v4_reconstruction_search.json"))
    return parser.parse_args()


def replace_magnitude(
    channel: torch.Tensor, target_power: torch.Tensor, dim: int, relaxation: float
) -> torch.Tensor:
    transformed = torch.fft.fft(channel, dim=dim, norm="ortho")
    magnitude = torch.abs(transformed)
    phase = transformed / magnitude.clamp_min(1e-20)
    desired = torch.sqrt(target_power.clamp_min(0.0))
    return torch.fft.ifft(magnitude.lerp(desired, relaxation) * phase, dim=dim, norm="ortho")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    data = RoundData(".")
    data.validate()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    all_positions = np.asarray(data.train_positions, dtype=np.float32)
    all_contexts = np.load("artifacts/map_context.npz")["train"].astype(np.float32)
    all_features = np.asarray(np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32)
    valid_global = nonzero_feature_indices(all_features)
    inverse = np.full(len(all_features), -1, dtype=np.int64)
    inverse[valid_global] = np.arange(len(valid_global))
    positions = all_positions[valid_global]
    contexts = all_contexts[valid_global]
    features = torch.from_numpy(all_features[valid_global]).to(device)
    val_global = np.asarray(checkpoint["validation_indices"], dtype=np.int64)[: args.limit]
    train_global = np.asarray(checkpoint["train_indices"], dtype=np.int64)
    val_idx, train_idx = inverse[val_global], inverse[train_global]
    embeddings = metric_embeddings(positions, contexts)
    pas = predict_config(PAS_CONFIG, embeddings[PAS_CONFIG.metric], train_idx, val_idx, features)
    pdp = predict_config(PDP_CONFIG, embeddings[PDP_CONFIG.metric], train_idx, val_idx, features)
    compact = torch.cat((pas[:, :1024], pdp[:, 1024:]), dim=1)
    transition, boundary = graph_matrices(
        embeddings["xy_y0.75"], train_idx, val_idx, features, k=24, power=2.5, softening=0.0
    )
    alpha = 0.1
    compact = torch.linalg.solve(
        torch.eye(len(val_idx), device=device) - alpha * transition,
        (1.0 - alpha) * compact + alpha * boundary,
    )
    neighbor_local, neighbor_distance = nearest_neighbors(positions[val_idx], positions[train_idx], 16)
    neighbor_idx = train_idx[neighbor_local]
    weights_np = distance_weights(neighbor_distance, power=2.0).astype(np.float32)
    iterations = (1, 2, 5, 10, 20, 40, 80)
    relaxations = (0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0)
    totals = defaultdict(lambda: {"pas": 0.0, "pdp": 0.0, "error": 0.0, "energy": 0.0, "count": 0})
    channels = data.train_channels
    for start in range(0, len(val_idx), args.batch_size):
        stop = min(start + args.batch_size, len(val_idx))
        source = torch.from_numpy(
            np.array(channels[valid_global[neighbor_idx[start:stop]]], dtype=np.complex64, copy=True)
        ).to(device)
        weights = torch.from_numpy(weights_np[start:stop]).to(device)
        initial = torch.sum(source * weights[:, :, None, None, None], dim=1)
        target = torch.from_numpy(
            np.array(channels[val_global[start:stop]], dtype=np.complex64, copy=True)
        ).to(device)
        target_pas, target_pdp = spectral_targets_from_features(compact[start:stop], data.dims)
        for final in ("pas", "pdp"):
            for relaxation in relaxations:
                prediction = initial
                for iteration in range(1, max(iterations) + 1):
                    if final == "pdp":
                        prediction = replace_magnitude(prediction, target_pas, 1, relaxation)
                        prediction = replace_magnitude(prediction, target_pdp, -1, relaxation)
                    else:
                        prediction = replace_magnitude(prediction, target_pdp, -1, relaxation)
                        prediction = replace_magnitude(prediction, target_pas, 1, relaxation)
                    if iteration in iterations:
                        name = f"ap_{final}_i{iteration}_r{relaxation:g}"
                        scaled = prediction * 1e-7
                        metrics = score_components(scaled, target, data.dims)
                        batch = stop - start
                        totals[name]["pas"] += metrics["c1_pas"].item() * batch
                        totals[name]["pdp"] += metrics["c2_pdp"].item() * batch
                        totals[name]["error"] += torch.sum(torch.abs(scaled - target).square()).item()
                        totals[name]["energy"] += torch.sum(torch.abs(target).square()).item()
                        totals[name]["count"] += batch
        print(f"processed {stop}/{len(val_idx)}", flush=True)
    result = {}
    for name, value in totals.items():
        c1 = value["pas"] / value["count"]
        c2 = value["pdp"] / value["count"]
        c3 = value["error"] / value["energy"]
        result[name] = {"c1_pas": c1, "c2_pdp": c2, "c3_nmse": c3, "score": 0.4*c1+0.4*c2+0.2/(1+c3)}
    top = sorted(result.items(), key=lambda item: item[1]["score"], reverse=True)
    print(json.dumps(top[:30], indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"top": top}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
