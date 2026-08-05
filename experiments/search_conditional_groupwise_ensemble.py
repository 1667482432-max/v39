from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from experiments.search_groupwise_kriging_ensemble import (
    component_scores,
    group_score,
    mix_groups,
)
from experiments.search_improved_graph_gpu import graph_matrices
from experiments.search_kriging_ensemble_gpu import CONFIGS, predict_config
from experiments.search_spatial_kernels_gpu import metric_embeddings
from physical_ai.features import nonzero_feature_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-validate map-conditioned corrections to groupwise kernel weights"
    )
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--regularizations", type=float, nargs="+", default=(0.03, 0.1, 0.3, 1.0))
    parser.add_argument("--final-regularization", type=float, default=0.03)
    parser.add_argument("--condition-mode", choices=("basic", "rich"), default="basic")
    parser.add_argument(
        "--base", type=Path, default=Path("artifacts/groupwise_kriging_v2.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/conditional_groupwise.json")
    )
    return parser.parse_args()


def condition_features(
    positions: np.ndarray, contexts: np.ndarray, mode: str
) -> np.ndarray:
    basic = np.concatenate((positions[:, :2], contexts[:, :7]), axis=1).astype(np.float32)
    if mode == "basic":
        return basic
    patch_height = contexts[:, 103:128]
    patch_density = contexts[:, 128:153]
    patch_summary = np.column_stack(
        (
            patch_height.mean(1), patch_height.std(1), patch_height.max(1),
            patch_density.mean(1), patch_density.std(1), patch_density.max(1),
        )
    )
    distances, _ = cKDTree(positions[:, :2]).query(positions[:, :2], k=17, workers=-1)
    density_summary = np.log1p(distances[:, (1, 4, 8, 16)])
    return np.concatenate((basic, patch_summary, density_summary), axis=1).astype(np.float32)


def standardized_condition(
    raw: np.ndarray, train: np.ndarray, query: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    # Geometry plus the seven low-dimensional Physical-AI map summaries.  Keeping
    # this input deliberately compact makes the conditional correction identifiable
    # from only four spatial validation folds.
    mean = raw[train].mean(axis=0, keepdims=True)
    std = raw[train].std(axis=0, keepdims=True).clip(1e-3)
    train_x = (raw[train] - mean) / std
    query_x = (raw[query] - mean) / std
    # An intercept is unnecessary: the global base logits already supply it.
    return torch.from_numpy(train_x).cuda(), torch.from_numpy(query_x).cuda(), mean[0], std[0]


def conditional_mix(
    bank: torch.Tensor, base_weights: torch.Tensor, condition: torch.Tensor, correction: torch.Tensor
) -> torch.Tensor:
    logits = torch.log(base_weights.clamp_min(1e-7))[None] + condition @ correction
    weights = torch.softmax(logits, dim=1)
    return torch.einsum("qc,cqd->qd", weights, bank)


def train_group_correction(
    prediction: torch.Tensor,
    target: torch.Tensor,
    condition: torch.Tensor,
    base_weights: torch.Tensor,
    group: int,
    steps: int,
    learning_rate: float,
    regularization: float,
) -> torch.Tensor:
    if group < 4:
        bank = prediction[:, :, :1024].reshape(prediction.shape[0], -1, 256, 4)[
            :, :, :, group
        ]
        group_target = target[:, :1024].reshape(-1, 256, 4)[:, :, group]
    else:
        polarization, ue = divmod(group - 4, 4)
        bank = prediction[:, :, 1024:].reshape(prediction.shape[0], -1, 2, 4, 192)[
            :, :, polarization, ue
        ]
        group_target = target[:, 1024:].reshape(-1, 2, 4, 192)[:, polarization, ue]
    correction = torch.zeros(
        condition.shape[1], prediction.shape[0], device=prediction.device, requires_grad=True
    )
    optimizer = torch.optim.AdamW([correction], lr=learning_rate, weight_decay=0.0)
    for _ in range(steps):
        mixed = conditional_mix(bank, base_weights, condition, correction)
        score = torch.nn.functional.cosine_similarity(mixed, group_target, dim=1).mean()
        loss = 1.0 - score + regularization * correction.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return correction.detach()


def mix_conditional_groups(
    prediction: torch.Tensor,
    base: torch.Tensor,
    condition: torch.Tensor,
    corrections: list[torch.Tensor],
) -> torch.Tensor:
    pas_bank = prediction[:, :, :1024].reshape(prediction.shape[0], -1, 256, 4)
    pdp_bank = prediction[:, :, 1024:].reshape(prediction.shape[0], -1, 2, 4, 192)
    output_pas = torch.empty_like(pas_bank[0])
    output_pdp = torch.empty_like(pdp_bank[0])
    for ue in range(4):
        logits = torch.log(base[ue].clamp_min(1e-7))[None] + condition @ corrections[ue]
        weights = torch.softmax(logits, dim=1)
        output_pas[:, :, ue] = torch.einsum(
            "qc,cqm->qm", weights, pas_bank[:, :, :, ue]
        )
    for polarization in range(2):
        for ue in range(4):
            group = 4 + polarization * 4 + ue
            logits = torch.log(base[group].clamp_min(1e-7))[None] + condition @ corrections[group]
            weights = torch.softmax(logits, dim=1)
            output_pdp[:, polarization, ue] = torch.einsum(
                "qc,cqs->qs", weights, pdp_bank[:, :, polarization, ue]
            )
    return torch.cat((output_pas.flatten(1), output_pdp.flatten(1)), dim=1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(20260804)
    positions_all = np.load("Round1_Train_Pos.npy").astype(np.float32)
    contexts_all = np.load("artifacts/map_context.npz")["train"].astype(np.float32)
    features_all = np.asarray(
        np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32
    )
    valid = nonzero_feature_indices(features_all)
    inverse = np.full(len(features_all), -1, dtype=np.int64)
    inverse[valid] = np.arange(len(valid))
    positions, contexts = positions_all[valid], contexts_all[valid]
    raw_condition = condition_features(positions, contexts, args.condition_mode)
    features = torch.from_numpy(features_all[valid].copy()).cuda()
    embeddings = metric_embeddings(positions, contexts)
    base_artifact = json.loads(args.base.read_text(encoding="utf-8"))
    checkpoints = ["20260804", "101", "202", "303", "404"]
    fold_predictions: list[torch.Tensor] = []
    fold_targets: list[torch.Tensor] = []
    fold_indices: list[np.ndarray] = []
    for suffix in checkpoints:
        checkpoint = torch.load(
            f"artifacts/cv_noout_split{suffix}.pt", map_location="cpu", weights_only=False
        )
        val_global = np.asarray(checkpoint["validation_indices"], dtype=np.int64)[: args.limit]
        train_global = np.asarray(checkpoint["train_indices"], dtype=np.int64)
        query_idx, train_idx = inverse[val_global], inverse[train_global]
        direct = torch.stack(
            [
                predict_config(config, embeddings[config.metric], train_idx, query_idx, features)
                for config in CONFIGS
            ]
        )
        transition, boundary = graph_matrices(
            embeddings["xy_y0.75"], train_idx, query_idx, features, 24, 2.5, 0.0
        )
        matrix = torch.eye(len(query_idx), device="cuda") - 0.1 * transition
        right = 0.9 * direct + 0.1 * boundary[None]
        propagated = torch.linalg.solve(
            matrix, right.permute(1, 0, 2).reshape(len(query_idx), -1)
        ).reshape(len(query_idx), len(CONFIGS), -1).permute(1, 0, 2)
        fold_predictions.append(propagated)
        fold_targets.append(features[query_idx])
        fold_indices.append(query_idx)
        print(f"generated fold {suffix}", flush=True)

    results: dict[str, list[dict[str, object]]] = {}
    for regularization in args.regularizations:
        rows: list[dict[str, object]] = []
        for heldout in range(5):
            train_folds = [i for i in range(5) if i != heldout]
            train_prediction = torch.cat([fold_predictions[i] for i in train_folds], dim=1)
            train_target = torch.cat([fold_targets[i] for i in train_folds])
            train_idx = np.concatenate([fold_indices[i] for i in train_folds])
            train_x, test_x, condition_mean, condition_std = standardized_condition(
                raw_condition, train_idx, fold_indices[heldout]
            )
            base = torch.tensor(
                base_artifact["loo"][heldout]["weights"], dtype=torch.float32, device="cuda"
            )
            corrections = [
                train_group_correction(
                    train_prediction,
                    train_target,
                    train_x,
                    base[group],
                    group,
                    args.steps,
                    args.learning_rate,
                    regularization,
                )
                for group in range(12)
            ]
            baseline = mix_groups(fold_predictions[heldout], base)
            conditional = mix_conditional_groups(
                fold_predictions[heldout], base, test_x, corrections
            )
            row = {
                "fold": checkpoints[heldout],
                "baseline": component_scores(baseline, fold_targets[heldout]),
                "conditional": component_scores(conditional, fold_targets[heldout]),
                "correction_rms": float(torch.stack(corrections).square().mean().sqrt()),
                "condition_mean": condition_mean.tolist(),
                "condition_std": condition_std.tolist(),
                "pas_corrections": [value.cpu().tolist() for value in corrections[:4]],
            }
            rows.append(row)
            print(f"reg={regularization:g} {row}", flush=True)
        results[str(regularization)] = rows

    output = {
        "regularizations": results,
        "summary": {
            key: {
                "baseline": np.mean([row["baseline"] for row in rows], axis=0).tolist(),
                "conditional": np.mean([row["conditional"] for row in rows], axis=0).tolist(),
            }
            for key, rows in results.items()
        },
    }
    # Fit the deployable PAS-only correction on all out-of-fold predictions.
    # PDP is intentionally left at the stable global ensemble because every
    # conditional PDP setting was neutral or negative under held-out testing.
    final_prediction = torch.cat(fold_predictions, dim=1)
    final_target = torch.cat(fold_targets)
    final_idx = np.concatenate(fold_indices)
    condition_mean = raw_condition[final_idx].mean(axis=0)
    condition_std = raw_condition[final_idx].std(axis=0).clip(1e-3)
    final_condition = torch.from_numpy(
        (raw_condition[final_idx] - condition_mean) / condition_std
    ).cuda()
    final_base = torch.tensor(base_artifact["weights"], dtype=torch.float32, device="cuda")
    final_corrections = [
        train_group_correction(
            final_prediction,
            final_target,
            final_condition,
            final_base[group],
            group,
            args.steps,
            args.learning_rate,
            args.final_regularization,
        )
        for group in range(4)
    ]
    output["final"] = {
        "regularization": args.final_regularization,
        "condition_mode": args.condition_mode,
        "condition_mean": condition_mean.tolist(),
        "condition_std": condition_std.tolist(),
        "pas_corrections": [value.cpu().tolist() for value in final_corrections],
    }
    print(json.dumps(output["summary"], indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
