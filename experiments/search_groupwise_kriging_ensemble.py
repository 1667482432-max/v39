from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from physical_ai.features import nonzero_feature_indices
from experiments.search_improved_graph_gpu import graph_matrices
from experiments.search_kriging_ensemble_gpu import CONFIGS, Config, predict_config
from experiments.search_spatial_kernels_gpu import metric_embeddings


PAS_CONFIG = Config("xy_ctx-patch_s4", "exponential", 16, 0.5, 0.01, True)
PDP_CONFIG = Config("xy_ctx-patch_s4", "exponential", 24, 0.75, 0.05, False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOO groupwise kriging ensemble")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--output", type=Path, default=Path("artifacts/groupwise_kriging.json"))
    return parser.parse_args()


def group_score(prediction: torch.Tensor, target: torch.Tensor, group: int) -> torch.Tensor:
    if group < 4:
        p = prediction[:, :1024].reshape(-1, 256, 4)[:, :, group]
        t = target[:, :1024].reshape(-1, 256, 4)[:, :, group]
    else:
        local = group - 4
        polarization, ue = divmod(local, 4)
        p = prediction[:, 1024:].reshape(-1, 2, 4, 192)[:, polarization, ue]
        t = target[:, 1024:].reshape(-1, 2, 4, 192)[:, polarization, ue]
    return torch.nn.functional.cosine_similarity(p, t, dim=-1).mean()


def optimize_weights(
    prediction: torch.Tensor, target: torch.Tensor, steps: int
) -> tuple[torch.Tensor, list[float]]:
    weights = []
    scores = []
    for group in range(12):
        logits = torch.zeros(prediction.shape[0], device=prediction.device, requires_grad=True)
        optimizer = torch.optim.Adam([logits], lr=0.08)
        for _ in range(steps):
            weight = torch.softmax(logits, dim=0)
            mixed = torch.einsum("c,cqd->qd", weight, prediction)
            score = group_score(mixed, target, group)
            entropy = torch.sum(weight * torch.log(weight.clamp_min(1e-8)))
            loss = 1.0 - score + 2e-5 * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        final_weight = torch.softmax(logits.detach(), dim=0)
        mixed = torch.einsum("c,cqd->qd", final_weight, prediction)
        weights.append(final_weight)
        scores.append(float(group_score(mixed, target, group)))
    return torch.stack(weights), scores


def mix_groups(prediction: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    pas_bank = prediction[:, :, :1024].reshape(prediction.shape[0], -1, 256, 4)
    pdp_bank = prediction[:, :, 1024:].reshape(prediction.shape[0], -1, 2, 4, 192)
    output_pas = torch.empty_like(pas_bank[0])
    output_pdp = torch.empty_like(pdp_bank[0])
    for ue in range(4):
        output_pas[:, :, ue] = torch.einsum("c,cqm->qm", weights[ue], pas_bank[:, :, :, ue])
    for polarization in range(2):
        for ue in range(4):
            group = 4 + polarization * 4 + ue
            output_pdp[:, polarization, ue] = torch.einsum(
                "c,cqs->qs", weights[group], pdp_bank[:, :, polarization, ue]
            )
    return torch.cat((output_pas.flatten(1), output_pdp.flatten(1)), dim=1)


def component_scores(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    pas = torch.nn.functional.cosine_similarity(
        prediction[:, :1024].reshape(-1, 256, 4),
        target[:, :1024].reshape(-1, 256, 4),
        dim=1,
    ).mean()
    pdp = torch.nn.functional.cosine_similarity(
        prediction[:, 1024:].reshape(-1, 2, 4, 192),
        target[:, 1024:].reshape(-1, 2, 4, 192),
        dim=-1,
    ).mean()
    return float(pas), float(pdp)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    all_positions = np.load("Round1_Train_Pos.npy").astype(np.float32)
    all_contexts = np.load("artifacts/map_context.npz")["train"].astype(np.float32)
    all_features = np.asarray(np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32)
    valid = nonzero_feature_indices(all_features)
    inverse = np.full(len(all_features), -1, dtype=np.int64)
    inverse[valid] = np.arange(len(valid))
    positions, contexts = all_positions[valid], all_contexts[valid]
    features = torch.from_numpy(all_features[valid].copy()).to(device)
    embeddings = metric_embeddings(positions, contexts)
    checkpoints = ["20260804", "101", "202", "303", "404"]
    fold_predictions, fold_targets, fold_baselines = [], [], []
    for suffix in checkpoints:
        checkpoint = torch.load(
            f"artifacts/cv_noout_split{suffix}.pt", map_location="cpu", weights_only=False
        )
        val_global = np.asarray(checkpoint["validation_indices"], dtype=np.int64)[: args.limit]
        train_global = np.asarray(checkpoint["train_indices"], dtype=np.int64)
        query_idx, train_idx = inverse[val_global], inverse[train_global]
        direct = torch.stack(
            [predict_config(config, embeddings[config.metric], train_idx, query_idx, features) for config in CONFIGS]
        )
        transition, boundary = graph_matrices(
            embeddings["xy_y0.75"], train_idx, query_idx, features, 24, 2.5, 0.0
        )
        matrix = torch.eye(len(query_idx), device=device) - 0.1 * transition
        right = 0.9 * direct + 0.1 * boundary[None]
        propagated = torch.linalg.solve(
            matrix, right.permute(1, 0, 2).reshape(len(query_idx), -1)
        ).reshape(len(query_idx), len(CONFIGS), -1).permute(1, 0, 2)
        pas_index = CONFIGS.index(PAS_CONFIG)
        pdp_index = CONFIGS.index(PDP_CONFIG)
        baseline_direct = torch.cat(
            (direct[pas_index, :, :1024], direct[pdp_index, :, 1024:]), dim=1
        )
        baseline = torch.linalg.solve(matrix, 0.9 * baseline_direct + 0.1 * boundary)
        fold_predictions.append(propagated)
        fold_targets.append(features[query_idx])
        fold_baselines.append(baseline)
        print(f"generated fold {suffix}", flush=True)

    loo = []
    for heldout in range(5):
        train_prediction = torch.cat(
            [fold_predictions[i] for i in range(5) if i != heldout], dim=1
        )
        train_target = torch.cat([fold_targets[i] for i in range(5) if i != heldout])
        weights, _ = optimize_weights(train_prediction, train_target, args.steps)
        prediction = mix_groups(fold_predictions[heldout], weights)
        score = component_scores(prediction, fold_targets[heldout])
        baseline_score = component_scores(fold_baselines[heldout], fold_targets[heldout])
        loo.append({
            "fold": checkpoints[heldout],
            "ensemble": score,
            "baseline": baseline_score,
            "weights": weights.cpu().tolist(),
        })
        print("LOO", loo[-1], flush=True)

    all_prediction = torch.cat(fold_predictions, dim=1)
    all_target = torch.cat(fold_targets)
    weights, train_group_scores = optimize_weights(all_prediction, all_target, args.steps)
    result = {
        "configs": [config.name for config in CONFIGS],
        "weights": weights.cpu().tolist(),
        "train_group_scores": train_group_scores,
        "loo": loo,
        "loo_ensemble_mean": np.mean([row["ensemble"] for row in loo], axis=0).tolist(),
        "loo_baseline_mean": np.mean([row["baseline"] for row in loo], axis=0).tolist(),
    }
    print(json.dumps(result, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
