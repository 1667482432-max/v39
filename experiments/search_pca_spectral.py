from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.search_conditional_groupwise_ensemble import condition_features
from experiments.search_groupwise_kriging_ensemble import mix_groups
from experiments.search_improved_graph_gpu import graph_matrices
from experiments.search_kriging_ensemble_gpu import CONFIGS, predict_config
from experiments.search_spatial_kernels_gpu import metric_embeddings
from physical_ai.features import nonzero_feature_indices
from physical_ai.neighbors import nearest_neighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-validate low-rank spectral projection")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--ranks", type=int, nargs="+", default=(4, 8, 16, 32, 64, 128))
    parser.add_argument(
        "--blends", type=float, nargs="+", default=(0.1, 0.25, 0.5, 0.75, 1.0)
    )
    parser.add_argument(
        "--groupwise", type=Path, default=Path("artifacts/groupwise_kriging_v2.json")
    )
    parser.add_argument(
        "--conditional", type=Path, default=Path("artifacts/conditional_groupwise_lowreg.json")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/pca_spectral.json"))
    return parser.parse_args()


def select_group(value: torch.Tensor, group: int) -> torch.Tensor:
    if group < 4:
        return value[:, :1024].reshape(-1, 256, 4)[:, :, group]
    polarization, ue = divmod(group - 4, 4)
    return value[:, 1024:].reshape(-1, 2, 4, 192)[:, polarization, ue]


def conditional_mix_pas(
    bank: torch.Tensor,
    global_weights: torch.Tensor,
    condition: torch.Tensor,
    corrections: torch.Tensor,
) -> torch.Tensor:
    pas_bank = bank[:, :, :1024].reshape(len(CONFIGS), -1, 256, 4)
    pdp_bank = bank[:, :, 1024:].reshape(len(CONFIGS), -1, 2, 4, 192)
    output_pas = torch.empty_like(pas_bank[0])
    output_pdp = torch.empty_like(pdp_bank[0])
    for ue in range(4):
        logits = torch.log(global_weights[ue].clamp_min(1e-7))[None]
        logits = logits + condition @ corrections[ue]
        output_pas[:, :, ue] = torch.einsum(
            "qc,cqm->qm", torch.softmax(logits, dim=1), pas_bank[:, :, :, ue]
        )
    for polarization in range(2):
        for ue in range(4):
            group = 4 + polarization * 4 + ue
            output_pdp[:, polarization, ue] = torch.einsum(
                "c,cqs->qs", global_weights[group], pdp_bank[:, :, polarization, ue]
            )
    return torch.cat((output_pas.flatten(1), output_pdp.flatten(1)), dim=1)


def unit(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=1, keepdim=True).clamp_min(1e-12)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    positions_all = np.load("Round1_Train_Pos.npy").astype(np.float32)
    contexts_all = np.load("artifacts/map_context.npz")["train"].astype(np.float32)
    features_all = np.asarray(
        np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32
    )
    valid = nonzero_feature_indices(features_all)
    inverse = np.full(len(features_all), -1, dtype=np.int64)
    inverse[valid] = np.arange(len(valid))
    positions, contexts = positions_all[valid], contexts_all[valid]
    features = torch.from_numpy(features_all[valid].copy()).to(device)
    embeddings = metric_embeddings(positions, contexts)
    groupwise = json.loads(args.groupwise.read_text(encoding="utf-8"))
    conditional = json.loads(args.conditional.read_text(encoding="utf-8"))
    regularization_key = f'{float(conditional["final"]["regularization"]):g}'
    raw_condition = condition_features(
        positions, contexts, conditional["final"].get("condition_mode", "basic")
    )
    checkpoints = ["20260804", "101", "202", "303", "404"]
    fold_results: list[dict[str, object]] = []
    aggregate: dict[str, list[float]] = defaultdict(list)

    for suffix in checkpoints:
        checkpoint = torch.load(
            f"artifacts/cv_noout_split{suffix}.pt", map_location="cpu", weights_only=False
        )
        val_global = np.asarray(checkpoint["validation_indices"], dtype=np.int64)[: args.limit]
        train_global = np.asarray(checkpoint["train_indices"], dtype=np.int64)
        val_idx, train_idx = inverse[val_global], inverse[train_global]
        bank = torch.stack(
            [
                predict_config(config, embeddings[config.metric], train_idx, val_idx, features)
                for config in CONFIGS
            ]
        )
        transition, boundary = graph_matrices(
            embeddings["xy_y0.75"], train_idx, val_idx, features, 24, 2.5, 0.0
        )
        matrix = torch.eye(len(val_idx), device=device) - 0.1 * transition
        bank = torch.linalg.solve(
            matrix,
            (0.9 * bank + 0.1 * boundary[None])
            .permute(1, 0, 2)
            .reshape(len(val_idx), -1),
        ).reshape(len(val_idx), len(CONFIGS), -1).permute(1, 0, 2)
        fold_weights = torch.tensor(
            next(row for row in groupwise["loo"] if row["fold"] == suffix)["weights"],
            dtype=torch.float32,
            device=device,
        )
        fold_conditional = next(
            row
            for row in conditional["regularizations"][regularization_key]
            if row["fold"] == suffix
        )
        condition = torch.from_numpy(
            (
                raw_condition[val_idx]
                - np.asarray(fold_conditional["condition_mean"], dtype=np.float32)
            )
            / np.asarray(fold_conditional["condition_std"], dtype=np.float32)
        ).to(device)
        corrections = torch.tensor(
            fold_conditional["pas_corrections"], dtype=torch.float32, device=device
        )
        compact = conditional_mix_pas(bank, fold_weights, condition, corrections)
        local_neighbor, _ = nearest_neighbors(
            positions[val_idx, :2], positions[train_idx, :2], 32
        )
        neighbor_idx = train_idx[local_neighbor]

        totals: dict[str, list[float]] = defaultdict(list)
        for group in range(12):
            train_group = unit(select_group(features[train_idx], group))
            target = unit(select_group(features[val_idx], group))
            prediction = unit(select_group(compact, group))
            totals["baseline"].append(float((prediction * target).sum(1).mean()))
            if group < 4:
                candidate = bank[:, :, :1024].reshape(len(CONFIGS), -1, 256, 4)[
                    :, :, :, group
                ]
                neighbor_group = features[neighbor_idx, :1024].reshape(
                    len(val_idx), 32, 256, 4
                )[:, :, :, group]
            else:
                polarization, ue = divmod(group - 4, 4)
                candidate = bank[:, :, 1024:].reshape(
                    len(CONFIGS), -1, 2, 4, 192
                )[:, :, polarization, ue]
                neighbor_group = features[neighbor_idx, 1024:].reshape(
                    len(val_idx), 32, 2, 4, 192
                )[:, :, polarization, ue]
            candidate = candidate / torch.linalg.vector_norm(
                candidate, dim=2, keepdim=True
            ).clamp_min(1e-12)
            candidate_similarity = torch.sum(candidate * target[None], dim=2)
            totals["oracle_config"].append(
                float(candidate_similarity.max(dim=0).values.mean())
            )
            neighbor_group = neighbor_group / torch.linalg.vector_norm(
                neighbor_group, dim=2, keepdim=True
            ).clamp_min(1e-12)
            neighbor_similarity = torch.sum(neighbor_group * target[:, None], dim=2)
            totals["oracle_neighbor32"].append(
                float(neighbor_similarity.max(dim=1).values.mean())
            )
            mean = train_group.mean(0, keepdim=True)
            centered = train_group - mean
            covariance = centered.T @ centered / len(train_group)
            _, vectors = torch.linalg.eigh(covariance)
            vectors = vectors.flip(1)
            for rank in args.ranks:
                if rank >= vectors.shape[1]:
                    continue
                basis = vectors[:, :rank]
                projected = mean + (prediction - mean) @ basis @ basis.T
                projected = unit(projected.clamp_min(0.0))
                for blend in args.blends:
                    mixed = unit(((1.0 - blend) * prediction + blend * projected).clamp_min(0.0))
                    score = float((mixed * target).sum(1).mean())
                    totals[f"r{rank}_b{blend:g}"].append(score)

        row: dict[str, object] = {"fold": suffix, "pas": {}, "pdp": {}}
        for name, values in totals.items():
            pas = float(np.mean(values[:4]))
            pdp = float(np.mean(values[4:]))
            row["pas"][name] = pas
            row["pdp"][name] = pdp
            aggregate[f"pas_{name}"].append(pas)
            aggregate[f"pdp_{name}"].append(pdp)
        fold_results.append(row)
        best_pas = max(row["pas"].items(), key=lambda item: item[1])
        best_pdp = max(row["pdp"].items(), key=lambda item: item[1])
        print(f"fold={suffix} best_pas={best_pas} best_pdp={best_pdp}", flush=True)

    mean_results = {name: float(np.mean(values)) for name, values in aggregate.items()}
    output = {
        "folds": fold_results,
        "mean": mean_results,
        "top_pas": sorted(
            ((name, score) for name, score in mean_results.items() if name.startswith("pas_")),
            key=lambda item: item[1], reverse=True,
        )[:20],
        "top_pdp": sorted(
            ((name, score) for name, score in mean_results.items() if name.startswith("pdp_")),
            key=lambda item: item[1], reverse=True,
        )[:20],
    }
    print(json.dumps({"top_pas": output["top_pas"][:10], "top_pdp": output["top_pdp"][:10]}, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
