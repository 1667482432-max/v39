from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from physical_ai.data import RoundData
from physical_ai.features import SpectralFeatureLayout, spectral_targets_from_features
from physical_ai.metrics import StreamingScore
from physical_ai.model import MapConditionedKernel, interpolate_features
from physical_ai.neighbors import distance_weights, nearest_neighbors
from physical_ai.spectral import alternating_spectral_projection
from physical_ai.transductive import transductive_graph_features


def load_model(path: Path) -> tuple[MapConditionedKernel, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    model = MapConditionedKernel(
        state["position_mean"], state["position_std"], state["context_mean"], state["context_std"]
    )
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end V3 validation without zero outliers")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/cv_noout_split20260804.pt"))
    parser.add_argument("--amplitude", type=float, default=1e-6)
    args = parser.parse_args()
    data = RoundData(".")
    data.validate()
    model, checkpoint = load_model(args.checkpoint)
    positions_np = np.asarray(data.train_positions, dtype=np.float32)
    positions = torch.from_numpy(positions_np)
    contexts = torch.from_numpy(np.load("artifacts/map_context.npz")["train"].astype(np.float32))
    features_np = np.asarray(np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32)
    features = torch.from_numpy(features_np.copy())
    channels = data.train_channels
    layout = SpectralFeatureLayout.from_dimensions(data.dims)
    val_idx = np.asarray(checkpoint["validation_indices"], dtype=np.int64)
    train_idx = np.asarray(checkpoint["train_indices"], dtype=np.int64)
    local, distances = nearest_neighbors(positions_np[val_idx], positions_np[train_idx], 16)
    neighbor_idx = train_idx[local]
    base_weights_np = distance_weights(distances, power=2.0).astype(np.float32)
    base_weights = torch.from_numpy(base_weights_np)
    direct = np.einsum(
        "qk,qkd->qd", base_weights_np, features_np[neighbor_idx], optimize=True
    )
    graph = transductive_graph_features(
        positions_np[train_idx], positions_np[val_idx], features_np[train_idx], direct,
        k=8, power=2.0, alpha=0.25,
    )
    query = torch.from_numpy(val_idx).long()
    neighbor = torch.from_numpy(neighbor_idx).long()
    with torch.inference_mode():
        pas_w, pdp_w = model(
            positions[query], contexts[query], positions[neighbor], contexts[neighbor]
        )
        learned = interpolate_features(features[neighbor], pas_w, pdp_w, layout.pas_size).numpy()
    candidates = {
        "direct": direct,
        "graph": graph,
        "graph_neural10": 0.9 * graph + 0.1 * learned,
    }
    scores = {name: StreamingScore(data.dims) for name in candidates}
    for i, target_idx in enumerate(val_idx):
        source = torch.from_numpy(np.array(channels[neighbor_idx[i]], dtype=np.complex64, copy=True))
        initial = torch.sum(source * base_weights[i].view(-1, 1, 1, 1), dim=0)
        target = torch.from_numpy(np.array(channels[target_idx], dtype=np.complex64, copy=True)).unsqueeze(0)
        for name, prediction in candidates.items():
            pas, pdp = spectral_targets_from_features(torch.from_numpy(prediction[i]), data.dims)
            reconstructed = alternating_spectral_projection(
                initial, pas, pdp, iterations=20, relaxation=0.5, final_constraint="pdp"
            )
            scores[name].update((reconstructed * args.amplitude).unsqueeze(0), target)
        if (i + 1) % 20 == 0:
            print(f"processed {i + 1}/{len(val_idx)}", flush=True)
    result = {name: score.compute() for name, score in scores.items()}
    print(json.dumps(result, indent=2))
    Path("artifacts/v3_official_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
