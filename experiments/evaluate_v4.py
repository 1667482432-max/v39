from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from physical_ai.data import RoundData
from physical_ai.features import nonzero_feature_indices, spectral_targets_from_features
from physical_ai.metrics import StreamingScore
from physical_ai.neighbors import distance_weights, nearest_neighbors
from physical_ai.spectral import alternating_spectral_projection
from experiments.benchmark_synthesis_gpu import differentiable_synthesis
from experiments.search_improved_graph_gpu import graph_matrices
from experiments.search_kriging_ensemble_gpu import CONFIGS, Config, predict_config
from experiments.search_spatial_kernels_gpu import metric_embeddings


PAS_CONFIG = Config("xy_ctx-patch_s4", "exponential", 16, 0.5, 0.01, True)
PDP_CONFIG = Config("xy_ctx-patch_s4", "exponential", 24, 0.75, 0.05, False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end V4 official validation")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/cv_noout_split20260804.pt"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--optimization-steps", type=int, default=80)
    parser.add_argument("--output", type=Path, default=Path("artifacts/v4_official_eval.json"))
    return parser.parse_args()


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
    global_to_local = np.full(len(all_features), -1, dtype=np.int64)
    global_to_local[valid_global] = np.arange(len(valid_global))
    positions = all_positions[valid_global]
    contexts = all_contexts[valid_global]
    features_np = all_features[valid_global]
    features = torch.from_numpy(features_np).to(device)
    val_global = np.asarray(checkpoint["validation_indices"], dtype=np.int64)[: args.limit]
    train_global = np.asarray(checkpoint["train_indices"], dtype=np.int64)
    val_idx = global_to_local[val_global]
    train_idx = global_to_local[train_global]
    if np.any(val_idx < 0) or np.any(train_idx < 0):
        raise RuntimeError("Checkpoint split contains excluded zero-channel rows")
    embeddings = metric_embeddings(positions, contexts)
    pas_prediction = predict_config(PAS_CONFIG, embeddings[PAS_CONFIG.metric], train_idx, val_idx, features)
    pdp_prediction = predict_config(PDP_CONFIG, embeddings[PDP_CONFIG.metric], train_idx, val_idx, features)
    kriging = torch.cat((pas_prediction[:, :1024], pdp_prediction[:, 1024:]), dim=1)
    transition, boundary = graph_matrices(
        embeddings["xy_y0.75"], train_idx, val_idx, features,
        k=24, power=2.5, softening=0.0,
    )
    alpha = 0.1
    kriging_graph = torch.linalg.solve(
        torch.eye(len(val_idx), device=device) - alpha * transition,
        (1.0 - alpha) * kriging + alpha * boundary,
    )
    ensemble_meta = json.loads(Path("artifacts/kriging_ensemble_gpu.json").read_text(encoding="utf-8"))
    ensemble_sources = torch.stack(
        [predict_config(config, embeddings[config.metric], train_idx, val_idx, features) for config in CONFIGS]
    )
    pas_weights = torch.tensor(
        [ensemble_meta["learned"]["pas"]["weights"][config.name] for config in CONFIGS], device=device
    )
    pdp_weights = torch.tensor(
        [ensemble_meta["learned"]["pdp"]["weights"][config.name] for config in CONFIGS], device=device
    )
    ensemble_pas = torch.einsum("c,cqd->qd", pas_weights, ensemble_sources)
    ensemble_pdp = torch.einsum("c,cqd->qd", pdp_weights, ensemble_sources)
    ensemble = torch.cat((ensemble_pas[:, :1024], ensemble_pdp[:, 1024:]), dim=1)
    compact_candidates = {
        "kriging": kriging,
        "kriging_graph": kriging_graph,
        "kriging_ensemble": ensemble,
    }
    scores = {
        f"{name}_{reconstruction}": StreamingScore(data.dims)
        for name in compact_candidates
        for reconstruction in ("ap20", "opt80")
    }
    channels = data.train_channels
    neighbor_local, neighbor_distance = nearest_neighbors(positions[val_idx], positions[train_idx], 16)
    neighbor_idx = train_idx[neighbor_local]
    weights_np = distance_weights(neighbor_distance, power=2.0).astype(np.float32)
    for start in range(0, len(val_idx), args.batch_size):
        stop = min(start + args.batch_size, len(val_idx))
        source_global = valid_global[neighbor_idx[start:stop]]
        source = torch.from_numpy(
            np.array(channels[source_global], dtype=np.complex64, copy=True)
        ).to(device)
        weights = torch.from_numpy(weights_np[start:stop]).to(device)
        initial = torch.sum(source * weights[:, :, None, None, None], dim=1)
        target = torch.from_numpy(
            np.array(channels[val_global[start:stop]], dtype=np.complex64, copy=True)
        )
        for name, compact in compact_candidates.items():
            target_pas, target_pdp = spectral_targets_from_features(compact[start:stop], data.dims)
            ap = torch.stack(
                [
                    alternating_spectral_projection(
                        initial[i], target_pas[i], target_pdp[i], iterations=20,
                        relaxation=0.5, final_constraint="pdp",
                    )
                    for i in range(stop - start)
                ]
            )
            optimized = differentiable_synthesis(
                ap, target_pas, target_pdp, args.optimization_steps, 3e-2
            )
            scores[f"{name}_ap20"].update(ap.cpu() * 1e-7, target)
            scores[f"{name}_opt80"].update(optimized.cpu() * 1e-7, target)
        print(f"processed {stop}/{len(val_idx)}", flush=True)
    result = {name: accumulator.compute() for name, accumulator in scores.items()}
    print(json.dumps(result, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
