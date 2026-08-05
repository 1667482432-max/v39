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
        description="Strict LOFO search for observable per-UE complex calibration"
    )
    parser.add_argument(
        "--stats-pattern", default="artifacts/v37_ue_stats_split{fold}.npz"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/v37_ue_calibration.json")
    )
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/v37_ue_calibration.npz")
    )
    return parser.parse_args()


def features(stats: dict[str, np.ndarray], mode: str) -> np.ndarray:
    return scalar_calibration_features(
        stats["position"],
        stats["context"],
        stats["nearest_distance"],
        stats["final_pred_energy"],
        stats["pred_energy_pol_ue"],
        mode,
    )


def targets(stats: dict[str, np.ndarray]) -> np.ndarray:
    scale = stats["final_cross_ue"] / np.maximum(
        stats["final_pred_energy_ue"], 1e-30
    )
    return np.stack((scale.real, scale.imag), axis=-1)


def evaluate_nmse(stats: dict[str, np.ndarray], scale: np.ndarray) -> float:
    cross = stats["final_cross_ue"]
    prediction_energy = stats["final_pred_energy_ue"]
    error = (
        stats["target_energy"].sum()
        + np.sum(np.abs(scale) ** 2 * prediction_energy)
        - 2.0 * np.real(np.sum(np.conj(scale) * cross))
    )
    return float(error / stats["target_energy"].sum())


def fit_models(
    x: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fitted = [
        fit_weighted_ridge(x, target[:, ue], weight[:, ue], regularization)
        for ue in range(target.shape[1])
    ]
    return tuple(np.stack(items, axis=0) for items in zip(*fitted))


def predict_models(
    x: np.ndarray,
    coefficient: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    values = [
        ridge_prediction(x, coefficient[ue], mean[ue], std[ue])
        for ue in range(len(coefficient))
    ]
    real_imag = np.stack(values, axis=1)
    return real_imag[..., 0] + 1j * real_imag[..., 1]


def main() -> None:
    args = parse_args()
    stats = {
        fold: dict(np.load(args.stats_pattern.format(fold=fold))) for fold in FOLDS
    }
    modes = ("basic", "advanced", "advanced_rbf")
    regularizations = (1.0, 10.0, 100.0, 1000.0, 10000.0)
    strengths = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
    results: dict[str, list[dict[str, object]]] = {}
    for mode in modes:
        fold_features = {fold: features(stats[fold], mode) for fold in FOLDS}
        for regularization in regularizations:
            raw_predictions = {}
            scalar_baselines = {}
            ue_baselines = {}
            for heldout in FOLDS:
                heldout_ids = set(stats[heldout]["global_index"].tolist())
                train_rows = []
                train_targets = []
                train_weights = []
                train_cross = []
                for fold in FOLDS:
                    if fold == heldout:
                        continue
                    keep = np.array(
                        [index not in heldout_ids for index in stats[fold]["global_index"]]
                    )
                    train_rows.append(fold_features[fold][keep])
                    train_targets.append(targets(stats[fold])[keep])
                    train_weights.append(stats[fold]["final_pred_energy_ue"][keep])
                    train_cross.append(stats[fold]["final_cross_ue"][keep])
                train_x = np.concatenate(train_rows)
                train_y = np.concatenate(train_targets)
                train_weight = np.concatenate(train_weights)
                cross = np.concatenate(train_cross)
                coefficient, mean, std = fit_models(
                    train_x, train_y, train_weight, regularization
                )
                raw_predictions[heldout] = predict_models(
                    fold_features[heldout], coefficient, mean, std
                )
                scalar_baselines[heldout] = cross.sum() / np.maximum(
                    train_weight.sum(), 1e-30
                )
                ue_baselines[heldout] = cross.sum(axis=0) / np.maximum(
                    train_weight.sum(axis=0), 1e-30
                )
            for baseline_kind, baselines in (
                ("scalar", scalar_baselines),
                ("ue", ue_baselines),
            ):
                for strength in strengths:
                    rows = []
                    for heldout in FOLDS:
                        baseline = np.asarray(baselines[heldout])
                        scale = baseline + strength * (
                            raw_predictions[heldout] - baseline
                        )
                        if scale.ndim == 1:
                            scale = np.broadcast_to(
                                scale, stats[heldout]["final_cross_ue"].shape
                            )
                        rows.append(
                            {
                                "fold": heldout,
                                "nmse": evaluate_nmse(stats[heldout], scale),
                                "scale_abs_quantiles": np.quantile(
                                    np.abs(scale),
                                    (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0),
                                ).tolist(),
                            }
                        )
                    key = (
                        f"{mode}__r{regularization:g}__b{baseline_kind}__s{strength:g}"
                    )
                    results[key] = rows
    summary = {
        key: {"nmse": float(np.mean([row["nmse"] for row in rows]))}
        for key, rows in results.items()
    }
    top = sorted(summary.items(), key=lambda item: item[1]["nmse"])
    best_key = top[0][0]
    mode, regularization_text, baseline_text, strength_text = best_key.split("__")
    regularization = float(regularization_text[1:])
    baseline_kind = baseline_text[1:]
    strength = float(strength_text[1:])

    all_x = np.concatenate([features(stats[fold], mode) for fold in FOLDS])
    all_y = np.concatenate([targets(stats[fold]) for fold in FOLDS])
    all_weight = np.concatenate(
        [stats[fold]["final_pred_energy_ue"] for fold in FOLDS]
    )
    coefficient, mean, std = fit_models(
        all_x, all_y, all_weight, regularization
    )
    all_cross = np.concatenate([stats[fold]["final_cross_ue"] for fold in FOLDS])
    if baseline_kind == "scalar":
        global_scale = np.full(
            4, all_cross.sum() / np.maximum(all_weight.sum(), 1e-30)
        )
    else:
        global_scale = all_cross.sum(axis=0) / np.maximum(
            all_weight.sum(axis=0), 1e-30
        )
    np.savez_compressed(
        args.model,
        coefficient=coefficient,
        feature_mean=mean,
        feature_std=std,
        global_scale=np.column_stack((global_scale.real, global_scale.imag)),
        strength=np.array(strength),
        mode=np.array(mode),
        groups=np.array(4),
        baseline=np.array(baseline_kind),
    )
    output = {
        "best": {"name": best_key, **summary[best_key], "folds": results[best_key]},
        "top": top[:30],
        "summary": summary,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"best": output["best"], "top": top[:15]}, indent=2))


if __name__ == "__main__":
    main()
