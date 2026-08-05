from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.search_ue_calibration import (
    FOLDS,
    evaluate_nmse,
    features,
    fit_models,
    predict_models,
    targets,
)
from physical_ai.scalar_calibration import fit_weighted_ridge, ridge_prediction


def scalar_target(stats: dict[str, np.ndarray]) -> np.ndarray:
    scale = stats["final_cross"] / np.maximum(stats["final_pred_energy"], 1e-30)
    return np.column_stack((scale.real, scale.imag))


def main() -> None:
    stats = {
        fold: dict(np.load(f"artifacts/v37_ue_stats_split{fold}.npz"))
        for fold in FOLDS
    }
    results: dict[str, list[dict[str, float | str]]] = {}
    for mode in ("advanced", "advanced_rbf"):
        fold_features = {fold: features(stats[fold], mode) for fold in FOLDS}
        for residual_regularization in (100.0, 1000.0, 10000.0, 100000.0):
            predictions = {}
            for heldout in FOLDS:
                heldout_ids = set(stats[heldout]["global_index"].tolist())
                x_rows = []
                scalar_rows = []
                scalar_weights = []
                residual_rows = []
                ue_weights = []
                for fold in FOLDS:
                    if fold == heldout:
                        continue
                    keep = np.array(
                        [index not in heldout_ids for index in stats[fold]["global_index"]]
                    )
                    scalar = scalar_target(stats[fold])[keep]
                    x_rows.append(fold_features[fold][keep])
                    scalar_rows.append(scalar)
                    scalar_weights.append(stats[fold]["final_pred_energy"][keep])
                    residual_rows.append(targets(stats[fold])[keep] - scalar[:, None, :])
                    ue_weights.append(stats[fold]["final_pred_energy_ue"][keep])
                train_x = np.concatenate(x_rows)
                scalar_y = np.concatenate(scalar_rows)
                scalar_weight = np.concatenate(scalar_weights)
                scalar_coefficient, scalar_mean, scalar_std = fit_weighted_ridge(
                    train_x, scalar_y, scalar_weight, 1000.0
                )
                common_raw = ridge_prediction(
                    fold_features[heldout],
                    scalar_coefficient,
                    scalar_mean,
                    scalar_std,
                )
                common = common_raw[:, 0] + 1j * common_raw[:, 1]
                residual_coefficient, residual_mean, residual_std = fit_models(
                    train_x,
                    np.concatenate(residual_rows),
                    np.concatenate(ue_weights),
                    residual_regularization,
                )
                residual = predict_models(
                    fold_features[heldout],
                    residual_coefficient,
                    residual_mean,
                    residual_std,
                )
                predictions[heldout] = (common, residual)
            for strength in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
                rows = []
                for heldout in FOLDS:
                    common, residual = predictions[heldout]
                    scale = common[:, None] + strength * residual
                    rows.append(
                        {
                            "fold": heldout,
                            "nmse": evaluate_nmse(stats[heldout], scale),
                        }
                    )
                key = f"{mode}__rr{residual_regularization:g}__s{strength:g}"
                results[key] = rows
    summary = {
        key: {"nmse": float(np.mean([row["nmse"] for row in rows]))}
        for key, rows in results.items()
    }
    top = sorted(summary.items(), key=lambda item: item[1]["nmse"])
    output = {
        "best": {"name": top[0][0], **top[0][1], "folds": results[top[0][0]]},
        "top": top[:30],
        "summary": summary,
    }
    best_mode, residual_regularization_text, strength_text = top[0][0].split("__")
    residual_regularization = float(residual_regularization_text[2:])
    strength = float(strength_text[1:])
    all_x = np.concatenate([features(stats[fold], best_mode) for fold in FOLDS])
    all_residual = np.concatenate(
        [
            targets(stats[fold]) - scalar_target(stats[fold])[:, None, :]
            for fold in FOLDS
        ]
    )
    all_weight = np.concatenate(
        [stats[fold]["final_pred_energy_ue"] for fold in FOLDS]
    )
    coefficient, mean, std = fit_models(
        all_x, all_residual, all_weight, residual_regularization
    )
    np.savez_compressed(
        "artifacts/v37_ue_residual_calibration.npz",
        coefficient=coefficient,
        feature_mean=mean,
        feature_std=std,
        strength=np.array(strength),
        mode=np.array(best_mode),
        groups=np.array(4),
    )
    path = Path("artifacts/v37_hierarchical_ue_search.json")
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"best": output["best"], "top": top[:15]}, indent=2))


if __name__ == "__main__":
    main()
