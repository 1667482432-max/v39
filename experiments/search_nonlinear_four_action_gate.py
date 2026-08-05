from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from experiments.search_four_action_sample_gate import (
    ACTIONS,
    FOLDS,
    PATTERNS,
    action_metrics,
    choose_actions,
    fit_multi_ridge,
    predict_multi_ridge,
)
from experiments.search_joint_sample_spectral_gate import aggregate_training
from physical_ai.spectral_calibration import LocalSpectralCorrection


FEATURE_MODES = {
    "spectral": np.arange(41, 83),
    "basic_spectral": np.concatenate((np.arange(9), np.arange(41, 83))),
    "all": np.arange(83),
    "condition": np.arange(41),
}


def load_data() -> dict[str, dict[str, object]]:
    output = {}
    for fold in FOLDS:
        stats = {
            action: dict(np.load(pattern.format(fold=fold)))
            for action, pattern in PATTERNS.items()
        }
        spectral = dict(np.load(f"artifacts/v39_spectral_stats_split{fold}.npz"))
        correction = LocalSpectralCorrection.load(
            Path(f"artifacts/v39_gated_spectral_s010_split{fold}.npz")
        )
        features = correction.sample_features(
            torch.from_numpy(np.asarray(spectral["prediction"], dtype=np.float32)),
            spectral["position"],
            spectral["context"],
        ).astype(np.float64)

        def delta(action: str) -> np.ndarray:
            base, candidate = stats["none"], stats[action]

            def sample_nmse(item: dict[str, np.ndarray]) -> np.ndarray:
                return 1.0 - np.abs(item["final_cross"]) ** 2 / np.maximum(
                    item["final_pred_energy"] * item["target_energy"], 1e-30
                )

            return (
                0.4 * (candidate["final_pas"] - base["final_pas"])
                + 0.4 * (candidate["final_pdp"] - base["final_pdp"])
                + 0.2
                * (
                    1.0 / (1.0 + sample_nmse(candidate))
                    - 1.0 / (1.0 + sample_nmse(base))
                )
            )

        output[fold] = {
            "stats": stats,
            "ids": stats["none"]["global_index"],
            "features": features,
            "target": np.column_stack([delta(action) for action in ACTIONS[1:]]),
        }
    return output


def aggregate_multi(ids, features, target):
    x = None
    columns = []
    for index in range(target.shape[1]):
        current_x, current_y = aggregate_training(ids, features, target[:, index])
        if x is None:
            x = current_x
        columns.append(current_y)
    return x, np.column_stack(columns)


def main() -> None:
    data = load_data()
    cache = {}
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
        mean = train_x.mean(axis=0)
        std = np.maximum(train_x.std(axis=0), 1e-6)
        linear_model = fit_multi_ridge(train_x, train_y, "all", 100.0)
        linear = predict_multi_ridge(data[heldout]["features"], *linear_model)
        cache[heldout] = {
            "train_x": train_x,
            "train_y": train_y,
            "query_x": data[heldout]["features"],
            "mean": mean,
            "std": std,
            "linear": linear,
        }

    fractions = tuple(np.linspace(0.1, 0.9, 17))
    results = {}
    for feature_mode, columns in FEATURE_MODES.items():
        neighbor_counts = (8, 16, 24, 32, 48, 64, 96, 128)
        distances = {}
        indices = {}
        for fold in FOLDS:
            item = cache[fold]
            train = (item["train_x"][:, columns] - item["mean"][columns]) / item["std"][columns]
            query = (item["query_x"][:, columns] - item["mean"][columns]) / item["std"][columns]
            distance, local = cKDTree(train).query(
                query, k=max(neighbor_counts), workers=-1
            )
            distances[fold] = distance
            indices[fold] = local
        for neighbors in neighbor_counts:
            for power in (0.0, 0.5, 1.0, 2.0):
                for softening in (0.1, 1.0, 3.0):
                    nonlinear = {}
                    for fold in FOLDS:
                        distance = distances[fold][:, :neighbors]
                        local = indices[fold][:, :neighbors]
                        if power == 0.0:
                            weight = np.ones_like(distance)
                        else:
                            weight = (distance + softening) ** (-power)
                        weight /= weight.sum(axis=1, keepdims=True)
                        nonlinear[fold] = np.einsum(
                            "qk,qka->qa",
                            weight,
                            cache[fold]["train_y"][local],
                            optimize=True,
                        )
                    for blend in (0.25, 0.5, 0.75, 1.0):
                        predictions = {
                            fold: (1.0 - blend) * cache[fold]["linear"]
                            + blend * nonlinear[fold]
                            for fold in FOLDS
                        }
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
                            name = (
                                f"{feature_mode}__k{neighbors}__p{power:g}"
                                f"__e{softening:g}__b{blend:g}__f{fraction:g}"
                            )
                            results[name] = {
                                "score": float(np.mean([row["score"] for row in rows])),
                                "pas": float(np.mean([row["pas"] for row in rows])),
                                "pdp": float(np.mean([row["pdp"] for row in rows])),
                                "nmse": float(np.mean([row["nmse"] for row in rows])),
                                "folds": rows,
                            }
    top = sorted(results.items(), key=lambda item: item[1]["score"], reverse=True)
    best_name, best = top[0]
    output = {
        "best": {"name": best_name, **best},
        "top": [[name, value] for name, value in top[:30]],
    }
    Path("artifacts/v42_nonlinear_four_action_search.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"best": output["best"], "top": output["top"][:10]}, indent=2))


if __name__ == "__main__":
    main()
