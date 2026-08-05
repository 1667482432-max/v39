from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from physical_ai.features import nonzero_feature_indices
from experiments.search_groupwise_kriging_ensemble import group_score, mix_groups
from experiments.search_improved_graph_gpu import graph_matrices
from experiments.search_kriging_ensemble_gpu import CONFIGS, predict_config
from experiments.search_spatial_kernels_gpu import metric_embeddings


ALPHAS = (0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4)


def main() -> None:
    device = torch.device("cuda")
    meta = json.loads(Path("artifacts/groupwise_kriging_v2.json").read_text())
    all_positions = np.load("Round1_Train_Pos.npy").astype(np.float32)
    all_contexts = np.load("artifacts/map_context.npz")["train"].astype(np.float32)
    all_features = np.asarray(np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32)
    valid = nonzero_feature_indices(all_features)
    inverse = np.full(len(all_features), -1, dtype=np.int64); inverse[valid] = np.arange(len(valid))
    positions, contexts = all_positions[valid], all_contexts[valid]
    features = torch.from_numpy(all_features[valid].copy()).to(device)
    embeddings = metric_embeddings(positions, contexts)
    suffixes = ["20260804", "101", "202", "303", "404"]
    fold_scores = []
    for suffix in suffixes:
        checkpoint = torch.load(f"artifacts/cv_noout_split{suffix}.pt", map_location="cpu", weights_only=False)
        val_global = np.asarray(checkpoint["validation_indices"], dtype=np.int64)[:200]
        train_global = np.asarray(checkpoint["train_indices"], dtype=np.int64)
        query_idx, train_idx = inverse[val_global], inverse[train_global]
        bank = torch.stack([predict_config(c, embeddings[c.metric], train_idx, query_idx, features) for c in CONFIGS])
        fold_meta = next(row for row in meta["loo"] if row["fold"] == suffix)
        weights = torch.tensor(fold_meta["weights"], device=device)
        direct = mix_groups(bank, weights)
        transition, boundary = graph_matrices(embeddings["xy_y0.75"], train_idx, query_idx, features, 24, 2.5, 0.0)
        identity = torch.eye(len(query_idx), device=device)
        scores = np.empty((len(ALPHAS), 12))
        for ai, alpha in enumerate(ALPHAS):
            prediction = torch.linalg.solve(
                identity - alpha * transition,
                (1.0 - alpha) * direct + alpha * boundary,
            )
            for group in range(12): scores[ai, group] = float(group_score(prediction, features[query_idx], group))
        fold_scores.append(scores)
        print("fold", suffix, "best", [ALPHAS[i] for i in np.argmax(scores, axis=0)], flush=True)
    fold_scores = np.stack(fold_scores)
    heldout_groups = []
    selections = []
    for heldout in range(5):
        train = np.mean(np.delete(fold_scores, heldout, axis=0), axis=0)
        selected = np.argmax(train, axis=0)
        selections.append([ALPHAS[i] for i in selected])
        heldout_groups.append(fold_scores[heldout, selected, np.arange(12)])
    heldout_groups = np.stack(heldout_groups)
    final_selected = np.argmax(np.mean(fold_scores, axis=0), axis=0)
    result = {
        "alphas": list(ALPHAS),
        "loo_selections": selections,
        "loo_group_scores": heldout_groups.tolist(),
        "loo_pas": float(heldout_groups[:, :4].mean()),
        "loo_pdp": float(heldout_groups[:, 4:].mean()),
        "fixed_pas": float(fold_scores[:, ALPHAS.index(0.1), :4].mean()),
        "fixed_pdp": float(fold_scores[:, ALPHAS.index(0.1), 4:].mean()),
        "final_alphas": [ALPHAS[i] for i in final_selected],
    }
    print(json.dumps(result, indent=2))
    Path("artifacts/groupwise_graph_alpha.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__": main()
