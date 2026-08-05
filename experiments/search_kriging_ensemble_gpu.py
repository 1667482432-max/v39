from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from physical_ai.features import nonzero_feature_indices
from experiments.search_spatial_kernels_gpu import metric_embeddings
from experiments.search_local_kriging_gpu import covariance


@dataclass(frozen=True)
class Config:
    metric: str
    kind: str
    k: int
    scale: float
    nugget: float
    positive: bool

    @property
    def name(self) -> str:
        mode = "pos" if self.positive else "raw"
        return f"{self.metric}_{self.kind}_k{self.k}_s{self.scale:g}_n{self.nugget:g}_{mode}"


CONFIGS = (
    Config("xy_ctx-patch_s4", "exponential", 16, 0.5, 0.001, False),
    Config("xy_ctx-patch_s4", "exponential", 16, 0.5, 0.01, False),
    Config("xy_ctx-patch_s4", "exponential", 16, 0.5, 0.05, False),
    Config("xy_ctx-patch_s4", "exponential", 16, 0.5, 0.01, True),
    Config("xy_ctx-patch_s4", "exponential", 24, 0.5, 0.05, False),
    Config("xy_ctx-patch_s4", "exponential", 32, 0.5, 0.05, False),
    Config("xy_ctx-patch_s4", "exponential", 16, 0.75, 0.05, False),
    Config("xy_ctx-patch_s4", "exponential", 24, 0.75, 0.05, False),
    Config("xy_ctx-patch_s4", "exponential", 32, 0.75, 0.05, False),
    Config("xy_ctx-patch_s4", "exponential", 16, 1.0, 0.05, False),
    Config("xy_ctx-patch_s4", "exponential", 16, 1.5, 0.05, False),
    Config("xy_ctx-patch_s4", "matern32", 24, 0.75, 0.001, True),
    Config("xy_ctx-all_s4", "exponential", 32, 0.5, 0.1, False),
    Config("xy_ctx-all_s4", "exponential", 32, 1.5, 0.05, False),
    Config("xy_ctx-summary_s4", "exponential", 24, 0.75, 0.05, False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convex ensemble of local kriging predictors")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260804, 101, 202, 303, 404])
    parser.add_argument("--validation-size", type=int, default=400)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("artifacts/kriging_ensemble_gpu.json"))
    return parser.parse_args()


def predict_config(
    config: Config,
    embedding: np.ndarray,
    train_idx: np.ndarray,
    query_idx: np.ndarray,
    features: torch.Tensor,
) -> torch.Tensor:
    distance_np, local = cKDTree(embedding[train_idx]).query(
        embedding[query_idx], k=config.k, workers=-1
    )
    neighbor_np = train_idx[local]
    neighbor_embedding = embedding[neighbor_np]
    pair_np = np.linalg.norm(
        neighbor_embedding[:, :, None, :] - neighbor_embedding[:, None, :, :], axis=-1
    ).astype(np.float32)
    device = features.device
    pair = torch.from_numpy(pair_np).to(device)
    distance = torch.from_numpy(distance_np.astype(np.float32)).to(device)
    bandwidth = distance[:, -1:, None] * config.scale
    cov_nn = covariance(pair, bandwidth, config.kind)
    cov_q = covariance(distance, bandwidth[:, :, 0], config.kind)
    q, k = distance.shape
    system = torch.zeros((q, k + 1, k + 1), device=device)
    system[:, :k, :k] = cov_nn + torch.eye(k, device=device) * config.nugget
    system[:, :k, k] = 1.0
    system[:, k, :k] = 1.0
    right = torch.cat((cov_q, torch.ones((q, 1), device=device)), dim=1)
    weight = torch.linalg.solve(system, right)[:, :k]
    if config.positive:
        weight = weight.clamp_min(0.0)
        weight /= weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
    neighbor = torch.from_numpy(neighbor_np.astype(np.int64)).to(device)
    return torch.einsum("qk,qkd->qd", weight, features[neighbor]).clamp_min(0.0)


def component_score(prediction: torch.Tensor, target: torch.Tensor, component: str) -> torch.Tensor:
    if component == "pas":
        return torch.nn.functional.cosine_similarity(
            prediction[:, :1024].reshape(-1, 256, 4),
            target[:, :1024].reshape(-1, 256, 4),
            dim=1,
        ).mean()
    return torch.nn.functional.cosine_similarity(
        prediction[:, 1024:].reshape(-1, 2, 4, 192),
        target[:, 1024:].reshape(-1, 2, 4, 192),
        dim=-1,
    ).mean()


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
    fold_predictions: list[torch.Tensor] = []
    fold_targets: list[torch.Tensor] = []
    fold_slices = []
    offset = 0
    for seed in args.seeds:
        permutation = np.random.default_rng(seed).permutation(len(features_np))
        query_idx = np.sort(permutation[: args.validation_size])
        train_idx = np.sort(permutation[args.validation_size :])
        predictions = torch.stack(
            [predict_config(config, embeddings[config.metric], train_idx, query_idx, features) for config in CONFIGS]
        )
        fold_predictions.append(predictions)
        fold_targets.append(features[query_idx])
        fold_slices.append((str(seed), offset, offset + len(query_idx)))
        offset += len(query_idx)
        print(f"generated fold={seed}", flush=True)
    predictions = torch.cat(fold_predictions, dim=1)
    targets = torch.cat(fold_targets, dim=0)
    individual = {}
    for index, config in enumerate(CONFIGS):
        pas = component_score(predictions[index], targets, "pas").item()
        pdp = component_score(predictions[index], targets, "pdp").item()
        individual[config.name] = {"pas": pas, "pdp": pdp, "mean": 0.5 * (pas + pdp)}
    learned = {}
    for component in ("pas", "pdp"):
        logits = torch.zeros(len(CONFIGS), device=device, requires_grad=True)
        optimizer = torch.optim.Adam([logits], lr=0.08)
        for _ in range(args.steps):
            weights = torch.softmax(logits, dim=0)
            mixed = torch.einsum("c,cqd->qd", weights, predictions)
            score = component_score(mixed, targets, component)
            entropy_penalty = 1e-5 * torch.sum(weights * torch.log(weights.clamp_min(1e-8)))
            loss = 1.0 - score + entropy_penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        weights = torch.softmax(logits.detach(), dim=0)
        mixed = torch.einsum("c,cqd->qd", weights, predictions)
        learned[component] = {
            "score": component_score(mixed, targets, component).item(),
            "weights": {CONFIGS[i].name: float(weights[i]) for i in range(len(CONFIGS))},
        }
    pas_weights = torch.tensor(
        [learned["pas"]["weights"][config.name] for config in CONFIGS], device=device
    )
    pdp_weights = torch.tensor(
        [learned["pdp"]["weights"][config.name] for config in CONFIGS], device=device
    )
    pas_mix = torch.einsum("c,cqd->qd", pas_weights, predictions)
    pdp_mix = torch.einsum("c,cqd->qd", pdp_weights, predictions)
    combined = torch.cat((pas_mix[:, :1024], pdp_mix[:, 1024:]), dim=1)
    folds = {}
    for seed, start, stop in fold_slices:
        pas = component_score(combined[start:stop], targets[start:stop], "pas").item()
        pdp = component_score(combined[start:stop], targets[start:stop], "pdp").item()
        folds[seed] = {"pas": pas, "pdp": pdp, "mean": 0.5 * (pas + pdp)}
    result = {"individual": individual, "learned": learned, "folds": folds}
    print(json.dumps(result, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
