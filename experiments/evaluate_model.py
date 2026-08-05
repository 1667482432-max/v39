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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate learned kernels with official metrics")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--features", type=Path, default=Path("artifacts/spectral_features.npy"))
    parser.add_argument("--context", type=Path, default=Path("artifacts/map_context.npz"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.pt"))
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("artifacts/model_official_eval.json"))
    return parser.parse_args()


def compact_cos(prediction: torch.Tensor, target: torch.Tensor, layout: SpectralFeatureLayout) -> tuple[float, float]:
    pp = prediction[:, : layout.pas_size].reshape(-1, 256, 4)
    tp = target[:, : layout.pas_size].reshape(-1, 256, 4)
    pd = prediction[:, layout.pas_size :].reshape(-1, 2, 4, 192)
    td = target[:, layout.pas_size :].reshape(-1, 2, 4, 192)
    return (
        torch.nn.functional.cosine_similarity(pp, tp, dim=1).mean().item(),
        torch.nn.functional.cosine_similarity(pd, td, dim=-1).mean().item(),
    )


def main() -> None:
    args = parse_args()
    data = RoundData(args.root)
    data.validate()
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    model = MapConditionedKernel(
        state["position_mean"], state["position_std"], state["context_mean"], state["context_std"]
    )
    model.load_state_dict(state)
    model.eval()
    positions = torch.from_numpy(np.asarray(data.train_positions, dtype=np.float32))
    contexts = torch.from_numpy(np.load(args.context)["train"].astype(np.float32))
    features = torch.from_numpy(np.asarray(np.load(args.features, mmap_mode="r"), dtype=np.float32).copy())
    channels = data.train_channels
    layout = SpectralFeatureLayout.from_dimensions(data.dims)
    val_idx = np.asarray(checkpoint["validation_indices"], dtype=np.int64)[: args.samples]
    train_idx = np.asarray(checkpoint["train_indices"], dtype=np.int64)
    query_np = np.asarray(data.train_positions)[val_idx]
    reference_np = np.asarray(data.train_positions)[train_idx]
    delta = query_np[:, None, :] - reference_np[None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    local = np.argsort(distance, axis=1)[:, : checkpoint["neighbors"]]
    neighbor_idx = train_idx[local]
    neighbor = torch.from_numpy(neighbor_idx).long()
    query = torch.from_numpy(val_idx).long()
    # A shallow graph denoiser is validation-positive only for PDP. Build it
    # strictly from the reference fold so validation targets cannot leak in.
    graph_local, graph_distance = nearest_neighbors(reference_np, reference_np, 5)
    graph_local, graph_distance = graph_local[:, 1:], graph_distance[:, 1:]
    graph_weights = torch.from_numpy(
        distance_weights(graph_distance, power=2.0).astype(np.float32)
    )
    graph_bank = features.clone()
    graph_prediction = torch.einsum(
        "qk,qkd->qd", graph_weights, features[torch.from_numpy(train_idx[graph_local]).long()]
    )
    graph_bank[torch.from_numpy(train_idx).long(), layout.pas_size :] = (
        0.9 * features[torch.from_numpy(train_idx).long(), layout.pas_size :]
        + 0.1 * graph_prediction[:, layout.pas_size :]
    )
    with torch.inference_mode():
        pas_w, pdp_w = model(
            positions[query], contexts[query], positions[neighbor], contexts[neighbor]
        )
        learned = interpolate_features(features[neighbor], pas_w, pdp_w, layout.pas_size)
        learned_graph = interpolate_features(graph_bank[neighbor], pas_w, pdp_w, layout.pas_size)
    d = np.take_along_axis(distance, local, axis=1)
    base_w_np = np.maximum(d, 1e-6) ** -2
    base_w_np /= base_w_np.sum(axis=1, keepdims=True)
    base_w = torch.from_numpy(base_w_np.astype(np.float32))
    baseline = torch.einsum("qk,qkd->qd", base_w, features[neighbor])
    baseline_graph = torch.einsum("qk,qkd->qd", base_w, graph_bank[neighbor])
    target_features = features[query]
    candidates = {
        "baseline": baseline,
        "baseline_graph": baseline_graph,
        "learned": learned,
        "learned_graph": learned_graph,
        "blend_half_graph": baseline.lerp(learned_graph, 0.5),
    }
    compact = {}
    for name, prediction in candidates.items():
        c1, c2 = compact_cos(prediction, target_features, layout)
        compact[name] = {"pas": c1, "pdp": c2, "mean": 0.5 * (c1 + c2)}
    print("compact", json.dumps(compact, indent=2))
    best_names = sorted(candidates, key=lambda x: compact[x]["mean"], reverse=True)[:3]
    accumulators = {name: StreamingScore(data.dims) for name in best_names}
    for i, target_idx in enumerate(val_idx):
        source_h = torch.from_numpy(np.array(channels[neighbor_idx[i]], dtype=np.complex64, copy=True))
        initial = torch.sum(source_h * base_w[i].view(-1, 1, 1, 1), dim=0)
        target = torch.from_numpy(np.array(channels[target_idx], dtype=np.complex64, copy=True)).unsqueeze(0)
        for name in best_names:
            predicted_feature = candidates[name][i]
            target_pas, target_pdp = spectral_targets_from_features(predicted_feature, data.dims)
            reconstructed = alternating_spectral_projection(
                initial, target_pas, target_pdp, iterations=20, relaxation=0.5, final_constraint="pdp"
            )
            accumulators[name].update(reconstructed.unsqueeze(0), target)
        if (i + 1) % 20 == 0:
            print(f"processed {i + 1}/{len(val_idx)}", flush=True)
    official = {blend: score.compute() for blend, score in accumulators.items()}
    print("official", json.dumps(official, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"compact": compact, "official": official}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
