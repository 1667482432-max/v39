from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from experiments.search_hierarchical_ue_calibration import scalar_target
from experiments.search_local_scalar_calibration import aggregate_reference
from experiments.search_ue_calibration import (
    FOLDS,
    evaluate_nmse,
    features,
    fit_models,
    predict_models,
    targets,
)
from physical_ai.scalar_calibration import fit_weighted_ridge, ridge_prediction
from physical_ai.spatial import metric_embeddings


def main() -> None:
    stats = {
        fold: dict(np.load(f"artifacts/v37_ue_stats_split{fold}.npz"))
        for fold in FOLDS
    }
    local_model = np.load("artifacts/v38_local_scalar_residual.npz")
    metric = str(local_model["metric"].item())
    neighbors = int(local_model["neighbors"].item())
    power = float(local_model["power"].item())
    softening = float(local_model["softening"].item())
    energy_gamma = float(local_model["energy_gamma"].item())
    selected_local_strength = float(local_model["strength"].item())

    offsets = {}
    cursor = 0
    for fold in FOLDS:
        offsets[fold] = np.arange(cursor, cursor + len(stats[fold]["position"]))
        cursor += len(stats[fold]["position"])
    all_positions = np.concatenate([stats[fold]["position"] for fold in FOLDS])
    all_contexts = np.concatenate([stats[fold]["context"] for fold in FOLDS])
    embedding = metric_embeddings(all_positions, all_contexts)[metric]
    fold_features = {fold: features(stats[fold], "advanced_rbf") for fold in FOLDS}
    predictions = {}
    for heldout in FOLDS:
        heldout_ids = set(stats[heldout]["global_index"].tolist())
        x_rows = []
        scalar_rows = []
        total_weights = []
        ue_residual_rows = []
        ue_weights = []
        train_ids = []
        embedding_rows = []
        for fold in FOLDS:
            if fold == heldout:
                continue
            keep = np.array(
                [index not in heldout_ids for index in stats[fold]["global_index"]]
            )
            scalar = scalar_target(stats[fold])[keep]
            x_rows.append(fold_features[fold][keep])
            scalar_rows.append(scalar)
            total_weights.append(stats[fold]["final_pred_energy"][keep])
            ue_residual_rows.append(targets(stats[fold])[keep] - scalar[:, None, :])
            ue_weights.append(stats[fold]["final_pred_energy_ue"][keep])
            train_ids.append(stats[fold]["global_index"][keep])
            embedding_rows.append(offsets[fold][keep])
        train_x = np.concatenate(x_rows)
        scalar_y = np.concatenate(scalar_rows)
        total_weight = np.concatenate(total_weights)
        reference_rows = np.concatenate(embedding_rows)
        coefficient, mean, std = fit_weighted_ridge(
            train_x, scalar_y, total_weight, 1000.0
        )
        base_train_ri = ridge_prediction(train_x, coefficient, mean, std)
        base_query_ri = ridge_prediction(
            fold_features[heldout], coefficient, mean, std
        )
        base = base_query_ri[:, 0] + 1j * base_query_ri[:, 1]
        local_residual_ri = scalar_y - base_train_ri
        local_residual = local_residual_ri[:, 0] + 1j * local_residual_ri[:, 1]
        reference_rows, local_residual, local_weight = aggregate_reference(
            np.concatenate(train_ids),
            local_residual,
            total_weight,
            reference_rows,
        )
        distance, indices = cKDTree(embedding[reference_rows]).query(
            embedding[offsets[heldout]], k=neighbors, workers=-1
        )
        if neighbors == 1:
            distance = distance[:, None]
            indices = indices[:, None]
        weight = (distance + softening) ** (-power)
        if energy_gamma != 0.0:
            energy = local_weight[indices]
            energy /= np.maximum(np.median(energy, axis=1, keepdims=True), 1e-30)
            weight *= energy**energy_gamma
        weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-30)
        local = np.sum(weight * local_residual[indices], axis=1)

        ue_coefficient, ue_mean, ue_std = fit_models(
            train_x,
            np.concatenate(ue_residual_rows),
            np.concatenate(ue_weights),
            100.0,
        )
        ue_residual = predict_models(
            fold_features[heldout], ue_coefficient, ue_mean, ue_std
        )
        predictions[heldout] = (base, local, ue_residual)

    results = {}
    local_strengths = sorted(
        set((0.15, 0.2, 0.25, 0.3, selected_local_strength))
    )
    for local_strength in local_strengths:
        for ue_strength in (0.0, 0.05, 0.1, 0.15, 0.2):
            rows = []
            for heldout in FOLDS:
                base, local, ue_residual = predictions[heldout]
                scale = (
                    base[:, None]
                    + local_strength * local[:, None]
                    + ue_strength * ue_residual
                )
                rows.append(
                    {
                        "fold": heldout,
                        "nmse": evaluate_nmse(stats[heldout], scale),
                    }
                )
            key = f"local{local_strength:g}__ue{ue_strength:g}"
            results[key] = rows
    summary = {
        key: {"nmse": float(np.mean([row["nmse"] for row in rows]))}
        for key, rows in results.items()
    }
    top = sorted(summary.items(), key=lambda item: item[1]["nmse"])
    best_key = top[0][0]
    best_local, best_ue = best_key.split("__")
    best_local_strength = float(best_local.removeprefix("local"))
    best_ue_strength = float(best_ue.removeprefix("ue"))
    output = {
        "local_metric": metric,
        "local_parameters": {
            "neighbors": neighbors,
            "power": power,
            "softening": softening,
            "energy_gamma": energy_gamma,
        },
        "best": {"name": best_key, **summary[best_key], "folds": results[best_key]},
        "top": top,
    }
    Path("artifacts/v38_combined_calibration.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    # Update only the two selected scalar strengths; all learned coefficients
    # and reference targets remain those fitted by their dedicated searches.
    local_payload = dict(np.load("artifacts/v38_local_scalar_residual.npz"))
    local_payload["strength"] = np.array(best_local_strength)
    np.savez_compressed("artifacts/v38_local_scalar_residual.npz", **local_payload)
    ue_payload = dict(np.load("artifacts/v37_ue_residual_calibration.npz"))
    ue_payload["strength"] = np.array(best_ue_strength)
    np.savez_compressed("artifacts/v38_ue_residual_calibration.npz", **ue_payload)
    print(json.dumps({"best": output["best"], "top": top[:10]}, indent=2))


if __name__ == "__main__":
    main()
