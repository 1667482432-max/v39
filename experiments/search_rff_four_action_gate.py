from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.search_four_action_sample_gate import (
    FOLDS,
    action_metrics,
    choose_actions,
)
from experiments.search_nonlinear_four_action_gate import (
    FEATURE_MODES,
    aggregate_multi,
    load_data,
)


def make_projection(width: int, output: int, gamma: float, seed: int):
    generator = np.random.default_rng(seed)
    weight = generator.normal(
        0.0, np.sqrt(2.0 * gamma / width), size=(width, output)
    )
    phase = generator.uniform(0.0, 2.0 * np.pi, size=output)
    return weight, phase


def transform(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    columns: np.ndarray,
    weight: np.ndarray,
    phase: np.ndarray,
    include_linear: bool,
) -> np.ndarray:
    normalized = (features[:, columns] - mean[columns]) / std[columns]
    nonlinear = np.sqrt(2.0 / weight.shape[1]) * np.cos(normalized @ weight + phase)
    if include_linear:
        return np.concatenate((normalized, nonlinear), axis=1)
    return nonlinear


def prepare_ridge(x: np.ndarray, target: np.ndarray):
    mean = target.mean(axis=0)
    eigenvalue, eigenvector = np.linalg.eigh(x.T @ x)
    projected = eigenvector.T @ x.T @ (target - mean)
    return eigenvalue, eigenvector, projected, mean


def ridge_coefficient(prepared, regularization: float) -> tuple[np.ndarray, np.ndarray]:
    eigenvalue, eigenvector, projected, mean = prepared
    coefficient = eigenvector @ (projected / (eigenvalue[:, None] + regularization))
    return coefficient, mean


def main() -> None:
    data = load_data()
    train_cache = {}
    for heldout in FOLDS:
        heldout_ids = set(data[heldout]["ids"].tolist())
        ids, x_rows, y_rows = [], [], []
        for fold in FOLDS:
            if fold == heldout:
                continue
            keep = np.asarray([index not in heldout_ids for index in data[fold]["ids"]])
            ids.append(data[fold]["ids"][keep])
            x_rows.append(data[fold]["features"][keep])
            y_rows.append(data[fold]["target"][keep])
        train_x, train_y = aggregate_multi(
            np.concatenate(ids), np.concatenate(x_rows), np.concatenate(y_rows)
        )
        train_cache[heldout] = {
            "x": train_x,
            "y": train_y,
            "mean": train_x.mean(axis=0),
            "std": np.maximum(train_x.std(axis=0), 1e-6),
        }

    fractions = tuple(np.linspace(0.1, 0.9, 17))
    results = {}
    for mode in ("condition", "basic_spectral", "all"):
        columns = FEATURE_MODES[mode]
        for output_width in (64, 128, 256):
            for gamma in (0.1, 0.3, 1.0, 3.0):
                for include_linear in (False, True):
                    predictions_by_reg = {
                        regularization: {} for regularization in (0.1, 1.0, 10.0, 100.0)
                    }
                    for fold in FOLDS:
                        item = train_cache[fold]
                        ensemble = []
                        for seed in (17, 43, 101):
                            weight, phase = make_projection(
                                len(columns), output_width, gamma, seed
                            )
                            train_phi = transform(
                                item["x"], item["mean"], item["std"], columns,
                                weight, phase, include_linear,
                            )
                            query_phi = transform(
                                data[fold]["features"], item["mean"], item["std"],
                                columns, weight, phase, include_linear,
                            )
                            ensemble.append(
                                (prepare_ridge(train_phi, item["y"]), query_phi)
                            )
                        for regularization in predictions_by_reg:
                            seed_predictions = []
                            for prepared, query_phi in ensemble:
                                coefficient, target_mean = ridge_coefficient(
                                    prepared, regularization
                                )
                                seed_predictions.append(
                                    query_phi @ coefficient + target_mean
                                )
                            predictions_by_reg[regularization][fold] = np.mean(
                                seed_predictions, axis=0
                            )
                    for regularization, predictions in predictions_by_reg.items():
                        for fraction in fractions:
                            rows = []
                            for fold in FOLDS:
                                action = choose_actions(predictions[fold], fraction)
                                rows.append(
                                    {
                                        "fold": fold,
                                        **action_metrics(data[fold]["stats"], action),
                                    }
                                )
                            transform_name = "linear_rff" if include_linear else "rff"
                            name = (
                                f"{mode}__{transform_name}__d{output_width}"
                                f"__g{gamma:g}__r{regularization:g}__f{fraction:g}"
                            )
                            results[name] = {
                                "score": float(np.mean([row["score"] for row in rows])),
                                "pas": float(np.mean([row["pas"] for row in rows])),
                                "pdp": float(np.mean([row["pdp"] for row in rows])),
                                "nmse": float(np.mean([row["nmse"] for row in rows])),
                                "folds": rows,
                            }

    top = sorted(results.items(), key=lambda item: item[1]["score"], reverse=True)
    output = {
        "best": {"name": top[0][0], **top[0][1]},
        "top": [[name, value] for name, value in top[:30]],
    }
    Path("artifacts/v43_rff_four_action_search.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"best": output["best"], "top": output["top"][:10]}, indent=2))


if __name__ == "__main__":
    main()
