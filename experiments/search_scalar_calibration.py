from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from physical_ai.scalar_calibration import (
    fit_weighted_ridge,
    ridge_prediction,
    scalar_calibration_features,
)


FOLDS = ("101", "202", "20260804", "303", "404")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LOFO search for observable per-sample complex calibration"
    )
    parser.add_argument(
        "--stats-pattern", default="artifacts/v37_scalar_stats_split{fold}.npz"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/v37_scalar_calibration.json")
    )
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/v37_scalar_calibration.npz")
    )
    return parser.parse_args()


def raw_features(stats: dict[str, np.ndarray], mode: str) -> np.ndarray:
    return scalar_calibration_features(
        stats["position"],
        stats["context"],
        stats["nearest_distance"],
        stats["final_pred_energy"],
        stats["pred_energy_pol_ue"],
        mode,
    )


def complex_target(stats: dict[str, np.ndarray]) -> np.ndarray:
    scale = stats["final_cross"] / np.maximum(stats["final_pred_energy"], 1e-30)
    return np.column_stack((scale.real, scale.imag))


def evaluate_nmse(stats: dict[str, np.ndarray], scale: np.ndarray) -> float:
    cross = stats["final_cross"]
    prediction_energy = stats["final_pred_energy"]
    target_energy = stats["target_energy"]
    error = (
        target_energy
        + np.abs(scale) ** 2 * prediction_energy
        - 2.0 * np.real(np.conj(scale) * cross)
    )
    return float(error.sum() / target_energy.sum())


def main() -> None:
    args = parse_args()
    stats = {
        fold: dict(np.load(args.stats_pattern.format(fold=fold))) for fold in FOLDS
    }
    modes = ("basic", "advanced", "advanced_rbf")
    regularizations = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    strengths = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
    results: dict[str, list[dict[str, object]]] = {}
    for mode in modes:
        fold_features = {fold: raw_features(stats[fold], mode) for fold in FOLDS}
        for regularization in regularizations:
            fold_models = {}
            raw_predictions = {}
            baselines = {}
            for heldout in FOLDS:
                heldout_ids = set(stats[heldout]["global_index"].tolist())
                train_rows = []
                train_targets = []
                train_weights = []
                for fold in FOLDS:
                    if fold == heldout:
                        continue
                    keep = np.array(
                        [index not in heldout_ids for index in stats[fold]["global_index"]]
                    )
                    train_rows.append(fold_features[fold][keep])
                    train_targets.append(complex_target(stats[fold])[keep])
                    train_weights.append(stats[fold]["final_pred_energy"][keep])
                train_x = np.concatenate(train_rows)
                train_y = np.concatenate(train_targets)
                train_weight = np.concatenate(train_weights)
                coefficient, mean, std = fit_weighted_ridge(
                    train_x, train_y, train_weight, regularization
                )
                raw = ridge_prediction(fold_features[heldout], coefficient, mean, std)
                raw_predictions[heldout] = raw[:, 0] + 1j * raw[:, 1]
                train_cross = sum(
                    stats[fold]["final_cross"][
                        np.array(
                            [index not in heldout_ids for index in stats[fold]["global_index"]]
                        )
                    ].sum()
                    for fold in FOLDS
                    if fold != heldout
                )
                train_energy = sum(
                    stats[fold]["final_pred_energy"][
                        np.array(
                            [index not in heldout_ids for index in stats[fold]["global_index"]]
                        )
                    ].sum()
                    for fold in FOLDS
                    if fold != heldout
                )
                baselines[heldout] = train_cross / max(train_energy, 1e-30)
                fold_models[heldout] = (coefficient, mean, std)
            for strength in strengths:
                rows = []
                for heldout in FOLDS:
                    baseline = baselines[heldout]
                    scale = baseline + strength * (raw_predictions[heldout] - baseline)
                    rows.append(
                        {
                            "fold": heldout,
                            "nmse": evaluate_nmse(stats[heldout], scale),
                            "baseline_nmse": evaluate_nmse(
                                stats[heldout],
                                np.full(len(scale), baseline, dtype=np.complex128),
                            ),
                            "scale_abs_quantiles": np.quantile(
                                np.abs(scale), (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
                            ).tolist(),
                            "negative_real_fraction": float(np.mean(scale.real < 0.0)),
                        }
                    )
                key = f"{mode}__r{regularization:g}__s{strength:g}"
                results[key] = rows
    summary = {
        key: {
            "nmse": float(np.mean([row["nmse"] for row in rows])),
            "baseline_nmse": float(
                np.mean([row["baseline_nmse"] for row in rows])
            ),
            "delta": float(
                np.mean([row["nmse"] - row["baseline_nmse"] for row in rows])
            ),
        }
        for key, rows in results.items()
    }
    top = sorted(summary.items(), key=lambda item: item[1]["nmse"])
    best_key = top[0][0]
    mode, regularization_text, strength_text = best_key.split("__")
    regularization = float(regularization_text[1:])
    strength = float(strength_text[1:])

    all_x = np.concatenate([raw_features(stats[fold], mode) for fold in FOLDS])
    all_y = np.concatenate([complex_target(stats[fold]) for fold in FOLDS])
    all_weight = np.concatenate([stats[fold]["final_pred_energy"] for fold in FOLDS])
    coefficient, mean, std = fit_weighted_ridge(
        all_x, all_y, all_weight, regularization
    )
    global_scale = sum(stats[fold]["final_cross"].sum() for fold in FOLDS) / sum(
        stats[fold]["final_pred_energy"].sum() for fold in FOLDS
    )
    np.savez_compressed(
        args.model,
        coefficient=coefficient,
        feature_mean=mean,
        feature_std=std,
        global_scale=np.array([global_scale.real, global_scale.imag]),
        strength=np.array(strength),
        mode=np.array(mode),
    )
    output = {
        "best": {"name": best_key, **summary[best_key], "folds": results[best_key]},
        "top": top[:30],
        "summary": summary,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"best": output["best"], "top": top[:10]}, indent=2))


if __name__ == "__main__":
    main()
