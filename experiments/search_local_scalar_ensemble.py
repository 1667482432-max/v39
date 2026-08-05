from __future__ import annotations

import argparse
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
from physical_ai.local_calibration import fit_metric_embedding
from physical_ai.spatial import ADVANCED_ENERGY_METRIC, ADVANCED_MAP_METRIC, metric_embeddings


COMPONENTS = (
    {
        "name": "endpoint",
        "metric": ADVANCED_ENERGY_METRIC,
        "neighbors": 24,
        "power": 2.0,
        "softening": 6.0,
        "energy_gamma": 1.0,
        "strength": 0.5,
    },
    {
        "name": "material",
        "metric": ADVANCED_MAP_METRIC,
        "neighbors": 24,
        "power": 0.5,
        "softening": 3.0,
        "energy_gamma": 1.0,
        "strength": 0.75,
    },
    {
        "name": "patch",
        "metric": "xy_ctx-patch_s4",
        "neighbors": 24,
        "power": 0.5,
        "softening": 3.0,
        "energy_gamma": 1.0,
        "strength": 0.75,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict LOFO local scalar and UE-residual calibration search"
    )
    parser.add_argument(
        "--stats-pattern", default="artifacts/v37_ue_stats_split{fold}.npz"
    )
    parser.add_argument(
        "--scalar-model", type=Path, default=Path("artifacts/v37_scalar_calibration.npz")
    )
    parser.add_argument(
        "--local-model",
        type=Path,
        default=Path("artifacts/v38_local_scalar_ensemble.npz"),
    )
    parser.add_argument(
        "--ue-model",
        type=Path,
        default=Path("artifacts/v38_ue_residual_calibration.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/v38_local_scalar_ensemble_search.json"),
    )
    return parser.parse_args()


def local_prediction(
    query_embedding: np.ndarray,
    reference_embedding: np.ndarray,
    residual: np.ndarray,
    reference_weight: np.ndarray,
    component: dict[str, object],
) -> np.ndarray:
    neighbors = int(component["neighbors"])
    distance, indices = cKDTree(reference_embedding).query(
        query_embedding, k=neighbors, workers=-1
    )
    weight = (distance + float(component["softening"])) ** (-float(component["power"]))
    energy = reference_weight[indices]
    energy /= np.maximum(np.median(energy, axis=1, keepdims=True), 1e-30)
    weight *= energy ** float(component["energy_gamma"])
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-30)
    return float(component["strength"]) * np.sum(weight * residual[indices], axis=1)


def main() -> None:
    args = parse_args()
    stats = {
        fold: dict(np.load(args.stats_pattern.format(fold=fold)))
        for fold in FOLDS
    }
    offsets = {}
    cursor = 0
    for fold in FOLDS:
        offsets[fold] = np.arange(cursor, cursor + len(stats[fold]["position"]))
        cursor += len(stats[fold]["position"])
    all_positions = np.concatenate([stats[fold]["position"] for fold in FOLDS])
    all_contexts = np.concatenate([stats[fold]["context"] for fold in FOLDS])
    embeddings = metric_embeddings(all_positions, all_contexts)
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
        coefficient, mean, std = fit_weighted_ridge(
            train_x, scalar_y, total_weight, 1000.0
        )
        base_train_ri = ridge_prediction(train_x, coefficient, mean, std)
        base_query_ri = ridge_prediction(
            fold_features[heldout], coefficient, mean, std
        )
        base = base_query_ri[:, 0] + 1j * base_query_ri[:, 1]
        residual_ri = scalar_y - base_train_ri
        residual = residual_ri[:, 0] + 1j * residual_ri[:, 1]
        reference_rows, residual, reference_weight = aggregate_reference(
            np.concatenate(train_ids),
            residual,
            total_weight,
            np.concatenate(embedding_rows),
        )
        local = np.stack(
            [
                local_prediction(
                    embeddings[item["metric"]][offsets[heldout]],
                    embeddings[item["metric"]][reference_rows],
                    residual,
                    reference_weight,
                    item,
                )
                for item in COMPONENTS
            ],
            axis=1,
        )
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
    grid = np.linspace(0.0, 1.0, 11)
    for first in grid:
        for second in grid:
            third = 1.0 - first - second
            if third < -1e-9:
                continue
            blend = np.array((first, second, max(third, 0.0)))
            for clip in (0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2):
                for ue_strength in (0.0, 0.05, 0.1, 0.15, 0.2):
                    rows = []
                    for heldout in FOLDS:
                        base, local, ue_residual = predictions[heldout]
                        correction = local @ blend
                        magnitude = np.abs(correction)
                        correction = correction * np.minimum(
                            1.0, clip / np.maximum(magnitude, 1e-30)
                        )
                        common = base + correction
                        scale = common[:, None] + ue_strength * ue_residual
                        rows.append(
                            {
                                "fold": heldout,
                                "nmse": evaluate_nmse(stats[heldout], scale),
                            }
                        )
                    clip_text = "inf" if np.isinf(clip) else f"{clip:g}"
                    key = (
                        f"w{first:g}_{second:g}_{max(third, 0.0):g}"
                        f"__c{clip_text}__ue{ue_strength:g}"
                    )
                    results[key] = rows
    summary = {
        key: {"nmse": float(np.mean([row["nmse"] for row in rows]))}
        for key, rows in results.items()
    }
    top = sorted(summary.items(), key=lambda item: item[1]["nmse"])
    best_key = top[0][0]
    weight_text, clip_text, ue_text = best_key.split("__")
    blend_weight = np.asarray(
        [float(value) for value in weight_text.removeprefix("w").split("_")]
    )
    clip = float(clip_text.removeprefix("c"))
    ue_strength = float(ue_text.removeprefix("ue"))

    all_x = np.concatenate([fold_features[fold] for fold in FOLDS])
    all_y = np.concatenate([scalar_target(stats[fold]) for fold in FOLDS])
    all_weight = np.concatenate(
        [stats[fold]["final_pred_energy"] for fold in FOLDS]
    )
    all_ids = np.concatenate([stats[fold]["global_index"] for fold in FOLDS])
    scalar_model = np.load(args.scalar_model)
    base_train = ridge_prediction(
        all_x,
        scalar_model["coefficient"],
        scalar_model["feature_mean"],
        scalar_model["feature_std"],
    )
    residual_ri = all_y - base_train
    residual = residual_ri[:, 0] + 1j * residual_ri[:, 1]
    reference_rows, residual, reference_weight = aggregate_reference(
        all_ids, residual, all_weight, np.arange(len(all_positions))
    )
    model_payload = {
        "reference_position": all_positions[reference_rows],
        "reference_context": all_contexts[reference_rows],
        "reference_residual": np.column_stack((residual.real, residual.imag)),
        "reference_weight": reference_weight,
        "metrics": np.asarray([item["metric"] for item in COMPONENTS]),
        "neighbors": np.asarray([item["neighbors"] for item in COMPONENTS]),
        "powers": np.asarray([item["power"] for item in COMPONENTS]),
        "softenings": np.asarray([item["softening"] for item in COMPONENTS]),
        "energy_gammas": np.asarray([item["energy_gamma"] for item in COMPONENTS]),
        "strengths": np.asarray([item["strength"] for item in COMPONENTS]),
        "blend_weight": blend_weight,
        "clip": np.array(clip),
    }
    for index, item in enumerate(COMPONENTS):
        fitted_embedding, mean, std, multiplier = fit_metric_embedding(
            all_positions, all_contexts, str(item["metric"])
        )
        np.testing.assert_allclose(
            fitted_embedding, embeddings[item["metric"]], rtol=1e-12, atol=1e-12
        )
        model_payload[f"reference_embedding_{index}"] = fitted_embedding[reference_rows]
        model_payload[f"context_mean_{index}"] = mean
        model_payload[f"context_std_{index}"] = std
        model_payload[f"context_multiplier_{index}"] = multiplier
    args.local_model.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.local_model, **model_payload)
    all_ue_residual = np.concatenate(
        [
            targets(stats[fold]) - scalar_target(stats[fold])[:, None, :]
            for fold in FOLDS
        ]
    )
    all_ue_weight = np.concatenate(
        [stats[fold]["final_pred_energy_ue"] for fold in FOLDS]
    )
    ue_coefficient, ue_mean, ue_std = fit_models(
        all_x, all_ue_residual, all_ue_weight, 100.0
    )
    args.ue_model.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.ue_model,
        coefficient=ue_coefficient,
        feature_mean=ue_mean,
        feature_std=ue_std,
        strength=np.array(ue_strength),
        mode=np.array("advanced_rbf"),
        groups=np.array(4),
    )
    output = {
        "components": COMPONENTS,
        "best": {"name": best_key, **summary[best_key], "folds": results[best_key]},
        "top": top[:50],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"best": output["best"], "top": top[:10]}, indent=2))


if __name__ == "__main__":
    main()
