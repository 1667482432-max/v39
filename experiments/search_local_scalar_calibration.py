from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from physical_ai.scalar_calibration import (
    fit_weighted_ridge,
    ridge_prediction,
    scalar_calibration_features,
)
from physical_ai.spatial import ADVANCED_ENERGY_METRIC, ADVANCED_MAP_METRIC, metric_embeddings


FOLDS = ("101", "202", "20260804", "303", "404")
METRICS = (
    "xy_ctx-patch_s4",
    ADVANCED_MAP_METRIC,
    ADVANCED_ENERGY_METRIC,
)


def features(stats: dict[str, np.ndarray]) -> np.ndarray:
    return scalar_calibration_features(
        stats["position"],
        stats["context"],
        stats["nearest_distance"],
        stats["final_pred_energy"],
        stats["pred_energy_pol_ue"],
        "advanced_rbf",
    )


def target(stats: dict[str, np.ndarray]) -> np.ndarray:
    scale = stats["final_cross"] / np.maximum(stats["final_pred_energy"], 1e-30)
    return np.column_stack((scale.real, scale.imag))


def evaluate_nmse(stats: dict[str, np.ndarray], scale: np.ndarray) -> float:
    error = (
        stats["target_energy"]
        + np.abs(scale) ** 2 * stats["final_pred_energy"]
        - 2.0 * np.real(np.conj(scale) * stats["final_cross"])
    )
    return float(error.sum() / stats["target_energy"].sum())


def aggregate_reference(
    global_index: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    embedding_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique, first, inverse, count = np.unique(
        global_index, return_index=True, return_inverse=True, return_counts=True
    )
    del unique
    total_weight = np.bincount(inverse, weights=weight)
    real = np.bincount(inverse, weights=weight * residual.real)
    imag = np.bincount(inverse, weights=weight * residual.imag)
    aggregated_residual = (real + 1j * imag) / np.maximum(total_weight, 1e-30)
    mean_weight = total_weight / count
    return embedding_rows[first], aggregated_residual, mean_weight


def main() -> None:
    stats = {
        fold: dict(np.load(f"artifacts/v37_scalar_stats_split{fold}.npz"))
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
    fold_features = {fold: features(stats[fold]) for fold in FOLDS}

    cache: dict[str, dict[str, object]] = {}
    for heldout in FOLDS:
        heldout_ids = set(stats[heldout]["global_index"].tolist())
        train_rows = []
        train_targets = []
        train_weights = []
        train_ids = []
        train_embedding_rows = []
        for fold in FOLDS:
            if fold == heldout:
                continue
            keep = np.array(
                [index not in heldout_ids for index in stats[fold]["global_index"]]
            )
            train_rows.append(fold_features[fold][keep])
            train_targets.append(target(stats[fold])[keep])
            train_weights.append(stats[fold]["final_pred_energy"][keep])
            train_ids.append(stats[fold]["global_index"][keep])
            train_embedding_rows.append(offsets[fold][keep])
        train_x = np.concatenate(train_rows)
        train_y = np.concatenate(train_targets)
        train_weight = np.concatenate(train_weights)
        embedding_rows = np.concatenate(train_embedding_rows)
        coefficient, mean, std = fit_weighted_ridge(
            train_x, train_y, train_weight, 1000.0
        )
        base_train = ridge_prediction(train_x, coefficient, mean, std)
        base_query_ri = ridge_prediction(
            fold_features[heldout], coefficient, mean, std
        )
        base_query = base_query_ri[:, 0] + 1j * base_query_ri[:, 1]
        residual_ri = train_y - base_train
        residual = residual_ri[:, 0] + 1j * residual_ri[:, 1]
        embedding_rows, residual, local_weight = aggregate_reference(
            np.concatenate(train_ids), residual, train_weight, embedding_rows
        )
        local = {}
        for metric in METRICS:
            distance, indices = cKDTree(embeddings[metric][embedding_rows]).query(
                embeddings[metric][offsets[heldout]], k=64, workers=-1
            )
            local[metric] = (distance, indices)
        cache[heldout] = {
            "base": base_query,
            "residual": residual,
            "weight": local_weight,
            "local": local,
        }

    results: dict[str, list[dict[str, float | str]]] = {}
    for metric in METRICS:
        for neighbors in (3, 4, 5, 6, 8, 24, 32, 40):
            for power in (0.5, 1.0, 1.5, 2.0):
                for softening in (1.0, 2.0, 3.0, 4.0, 6.0):
                    for energy_gamma in (0.5, 1.0, 1.5):
                        local_predictions = {}
                        for heldout in FOLDS:
                            item = cache[heldout]
                            distance, indices = item["local"][metric]
                            distance = distance[:, :neighbors]
                            indices = indices[:, :neighbors]
                            weight = (distance + softening) ** (-power)
                            if energy_gamma != 0.0:
                                energy = item["weight"][indices]
                                energy /= np.maximum(np.median(energy, axis=1, keepdims=True), 1e-30)
                                weight *= energy**energy_gamma
                            weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-30)
                            local_predictions[heldout] = np.sum(
                                weight * item["residual"][indices], axis=1
                            )
                        for strength in (0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75):
                            rows = []
                            for heldout in FOLDS:
                                scale = (
                                    cache[heldout]["base"]
                                    + strength * local_predictions[heldout]
                                )
                                rows.append(
                                    {
                                        "fold": heldout,
                                        "nmse": evaluate_nmse(stats[heldout], scale),
                                    }
                                )
                            key = (
                                f"{metric}__k{neighbors}__p{power:g}__e{softening:g}"
                                f"__g{energy_gamma:g}__s{strength:g}"
                            )
                            results[key] = rows
    summary = {
        key: {"nmse": float(np.mean([row["nmse"] for row in rows]))}
        for key, rows in results.items()
    }
    top = sorted(summary.items(), key=lambda item: item[1]["nmse"])
    best_key = top[0][0]
    metric, neighbors_text, power_text, softening_text, gamma_text, strength_text = (
        best_key.split("__")
    )
    neighbors = int(neighbors_text[1:])
    power = float(power_text[1:])
    softening = float(softening_text[1:])
    energy_gamma = float(gamma_text[1:])
    strength = float(strength_text[1:])

    all_x = np.concatenate([fold_features[fold] for fold in FOLDS])
    all_y = np.concatenate([target(stats[fold]) for fold in FOLDS])
    all_weight = np.concatenate(
        [stats[fold]["final_pred_energy"] for fold in FOLDS]
    )
    all_global_index = np.concatenate(
        [stats[fold]["global_index"] for fold in FOLDS]
    )
    scalar_model = np.load("artifacts/v37_scalar_calibration.npz")
    base_train = ridge_prediction(
        all_x,
        scalar_model["coefficient"],
        scalar_model["feature_mean"],
        scalar_model["feature_std"],
    )
    residual_ri = all_y - base_train
    residual = residual_ri[:, 0] + 1j * residual_ri[:, 1]
    reference_rows, residual, reference_weight = aggregate_reference(
        all_global_index,
        residual,
        all_weight,
        np.arange(len(all_positions)),
    )
    np.savez_compressed(
        "artifacts/v38_local_scalar_residual.npz",
        reference_position=all_positions[reference_rows],
        reference_context=all_contexts[reference_rows],
        reference_residual=np.column_stack((residual.real, residual.imag)),
        reference_weight=reference_weight,
        metric=np.array(metric),
        neighbors=np.array(neighbors),
        power=np.array(power),
        softening=np.array(softening),
        energy_gamma=np.array(energy_gamma),
        strength=np.array(strength),
    )
    output = {
        "baseline_nmse": 0.6162567841666013,
        "best": {"name": best_key, **summary[best_key], "folds": results[best_key]},
        "top": top[:50],
        "summary": summary,
    }
    Path("artifacts/v38_local_scalar_search.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"best": output["best"], "top": top[:15]}, indent=2))


if __name__ == "__main__":
    main()
