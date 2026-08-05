from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.benchmark_transductive import cosine_parts, graph_prediction
from physical_ai.data import RoundData
from physical_ai.features import SpectralFeatureLayout
from physical_ai.model import MapConditionedKernel, interpolate_features
from physical_ai.neighbors import distance_weights, nearest_neighbors


CHECKPOINTS = {
    20260804: Path("artifacts/cv_noout_split20260804.pt"),
    101: Path("artifacts/cv_noout_split101.pt"),
    202: Path("artifacts/cv_noout_split202.pt"),
    303: Path("artifacts/cv_noout_split303.pt"),
    404: Path("artifacts/cv_noout_split404.pt"),
}


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
    data = RoundData(".")
    data.validate()
    positions_np = np.asarray(data.train_positions, dtype=np.float32)
    positions = torch.from_numpy(positions_np)
    contexts = torch.from_numpy(np.load("artifacts/map_context.npz")["train"].astype(np.float32))
    features_np = np.asarray(np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32)
    features = torch.from_numpy(features_np.copy())
    layout = SpectralFeatureLayout.from_dimensions(data.dims)
    blends = (0.0, 0.1, 0.25, 0.5, 1.0)
    fold_results = {}
    for seed, path in CHECKPOINTS.items():
        model, checkpoint = load_model(path)
        val_idx = np.asarray(checkpoint["validation_indices"], dtype=np.int64)
        train_idx = np.asarray(checkpoint["train_indices"], dtype=np.int64)
        local, distances = nearest_neighbors(positions_np[val_idx], positions_np[train_idx], 16)
        neighbor_idx = train_idx[local]
        base_weights = distance_weights(distances, power=2.0).astype(np.float32)
        direct = np.einsum(
            "qk,qkd->qd", base_weights, features_np[neighbor_idx], optimize=True
        )
        transductive = graph_prediction(
            positions_np[train_idx],
            positions_np[val_idx],
            features_np[train_idx],
            direct,
            k=8,
            power=2.0,
            alpha=0.25,
        )
        query = torch.from_numpy(val_idx).long()
        neighbor = torch.from_numpy(neighbor_idx).long()
        with torch.inference_mode():
            pas_w, pdp_w = model(
                positions[query], contexts[query], positions[neighbor], contexts[neighbor]
            )
            learned = interpolate_features(features[neighbor], pas_w, pdp_w, layout.pas_size).numpy()
        fold = {}
        for blend in blends:
            prediction = transductive * (1.0 - blend) + learned * blend
            c1, c2 = cosine_parts(prediction, features_np[val_idx], layout)
            fold[str(blend)] = {"pas": c1, "pdp": c2, "mean": 0.5 * (c1 + c2)}
        fold_results[str(seed)] = fold
    summary = {}
    for blend in blends:
        key = str(blend)
        summary[key] = {
            metric: float(np.mean([fold_results[str(seed)][key][metric] for seed in CHECKPOINTS]))
            for metric in ("pas", "pdp", "mean")
        }
    print(json.dumps({"folds": fold_results, "summary": summary}, indent=2))
    Path("artifacts/stack_benchmark.json").write_text(
        json.dumps({"folds": fold_results, "summary": summary}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
