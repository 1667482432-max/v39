from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.search_joint_sample_spectral_gate import (
    FOLDS,
    aggregate_training,
    fit_ridge,
    sample_joint_delta,
    top_fraction_mask,
)
from physical_ai.spectral_calibration import LocalSpectralCorrection


ACTIONS = ("none", "pas", "pdp", "both")
PATTERNS = {
    "none": "artifacts/v40_nospec_full_stats_split{fold}.npz",
    "pas": "artifacts/v41_pas_only_full_stats_split{fold}.npz",
    "pdp": "artifacts/v41_pdp_only_full_stats_split{fold}.npz",
    "both": "artifacts/v39_s010_full_stats_split{fold}.npz",
}


def fit_multi_ridge(
    features: np.ndarray,
    target: np.ndarray,
    mode: str,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    models = [
        fit_ridge(features, target[:, index], mode, regularization)
        for index in range(target.shape[1])
    ]
    coefficient = np.stack([model[0] for model in models])
    return coefficient, models[0][1], models[0][2]


def predict_multi_ridge(
    features: np.ndarray,
    coefficient: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    normalized = (features - mean) / np.maximum(std, 1e-6)
    return normalized @ coefficient[:, :-1].T + coefficient[:, -1]


def choose_actions(value: np.ndarray, fraction: float) -> np.ndarray:
    best_action = np.argmax(value, axis=1).astype(np.int64) + 1
    benefit = np.max(value, axis=1)
    selected = top_fraction_mask(benefit, fraction)
    return np.where(selected, best_action, 0)


def action_metrics(
    stats: dict[str, dict[str, np.ndarray]], action: np.ndarray
) -> dict[str, float | dict[str, float]]:
    def select(name: str) -> np.ndarray:
        bank = np.stack([stats[item][name] for item in ACTIONS], axis=1)
        index = action.reshape((len(action), 1) + (1,) * (bank.ndim - 2))
        return np.take_along_axis(bank, index, axis=1).squeeze(1)

    pas = float(select("final_pas").mean())
    pdp = float(select("final_pdp").mean())
    cross = select("final_cross").sum()
    prediction_energy = select("final_pred_energy").sum()
    target_energy = stats["none"]["target_energy"].sum()
    nmse = float(
        1.0 - np.abs(cross) ** 2 / max(prediction_energy * target_energy, 1e-30)
    )
    fractions = {
        name: float(np.mean(action == index))
        for index, name in enumerate(ACTIONS)
    }
    return {
        "pas": pas,
        "pdp": pdp,
        "nmse": nmse,
        "score": 0.4 * pas + 0.4 * pdp + 0.2 / (1.0 + nmse),
        "action_fraction": fractions,
    }


def mixed_payload(
    stats: dict[str, dict[str, np.ndarray]], action: np.ndarray
) -> dict[str, np.ndarray]:
    output = {}
    base = stats["none"]
    for name, base_value in base.items():
        values = [stats[item].get(name) for item in ACTIONS]
        if (
            all(value is not None for value in values)
            and base_value.ndim > 0
            and all(value.shape == base_value.shape for value in values)
            and len(base_value) == len(action)
        ):
            bank = np.stack(values, axis=1)
            index = action.reshape((len(action), 1) + (1,) * (bank.ndim - 2))
            output[name] = np.take_along_axis(bank, index, axis=1).squeeze(1)
        else:
            output[name] = base_value
    output["sample_spectral_action"] = action
    return output


def main() -> None:
    fold_data = {}
    for fold in FOLDS:
        stats = {
            action: dict(np.load(pattern.format(fold=fold)))
            for action, pattern in PATTERNS.items()
        }
        for action in ACTIONS[1:]:
            np.testing.assert_array_equal(
                stats["none"]["global_index"], stats[action]["global_index"]
            )
        spectral = dict(np.load(f"artifacts/v39_spectral_stats_split{fold}.npz"))
        correction = LocalSpectralCorrection.load(
            Path(f"artifacts/v39_gated_spectral_s010_split{fold}.npz")
        )
        features = correction.sample_features(
            torch.from_numpy(np.asarray(spectral["prediction"], dtype=np.float32)),
            spectral["position"],
            spectral["context"],
        ).astype(np.float64)
        target = np.column_stack(
            [
                sample_joint_delta(stats["none"], stats[action])
                for action in ACTIONS[1:]
            ]
        )
        fold_data[fold] = {
            "stats": stats,
            "ids": stats["none"]["global_index"],
            "features": features,
            "target": target,
        }

    modes = ("spectral", "basic_spectral", "all")
    regularizations = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    fractions = tuple(np.linspace(0.0, 1.0, 21))
    predictions = {}
    fitted = {}
    results = {}
    v40_fold_score = {}
    for fold in FOLDS:
        v40 = dict(np.load(f"artifacts/v40_joint_full_stats_split{fold}.npz"))
        v40_fold_score[fold] = action_metrics(
            {
                "none": v40,
                "pas": v40,
                "pdp": v40,
                "both": v40,
            },
            np.zeros(len(v40["global_index"]), dtype=np.int64),
        )["score"]

    for mode in modes:
        for regularization in regularizations:
            key = (mode, regularization)
            predictions[key] = {}
            fitted[key] = {}
            for heldout in FOLDS:
                heldout_ids = set(fold_data[heldout]["ids"].tolist())
                ids, x_rows, y_rows = [], [], []
                for fold in FOLDS:
                    if fold == heldout:
                        continue
                    keep = np.asarray(
                        [index not in heldout_ids for index in fold_data[fold]["ids"]]
                    )
                    ids.append(fold_data[fold]["ids"][keep])
                    x_rows.append(fold_data[fold]["features"][keep])
                    y_rows.append(fold_data[fold]["target"][keep])
                train_ids = np.concatenate(ids)
                raw_x = np.concatenate(x_rows)
                raw_y = np.concatenate(y_rows)
                unique = np.unique(train_ids)
                aggregated_x = None
                aggregated_y = []
                for column in range(raw_y.shape[1]):
                    current_x, current_y = aggregate_training(
                        train_ids, raw_x, raw_y[:, column]
                    )
                    if aggregated_x is None:
                        aggregated_x = current_x
                    aggregated_y.append(current_y)
                del unique
                model = fit_multi_ridge(
                    aggregated_x,
                    np.column_stack(aggregated_y),
                    mode,
                    regularization,
                )
                fitted[key][heldout] = model
                predictions[key][heldout] = predict_multi_ridge(
                    fold_data[heldout]["features"], *model
                )
            for fraction in fractions:
                rows = []
                for heldout in FOLDS:
                    action = choose_actions(predictions[key][heldout], fraction)
                    rows.append(
                        {
                            "fold": heldout,
                            **action_metrics(fold_data[heldout]["stats"], action),
                        }
                    )
                name = f"{mode}__r{regularization:g}__f{fraction:g}"
                results[name] = {
                    "score": float(np.mean([row["score"] for row in rows])),
                    "pas": float(np.mean([row["pas"] for row in rows])),
                    "pdp": float(np.mean([row["pdp"] for row in rows])),
                    "nmse": float(np.mean([row["nmse"] for row in rows])),
                    "positive_vs_v40_folds": int(
                        sum(row["score"] > v40_fold_score[row["fold"]] for row in rows)
                    ),
                    "folds": rows,
                }

    top = sorted(results.items(), key=lambda item: item[1]["score"], reverse=True)
    stable = [item for item in top if item[1]["positive_vs_v40_folds"] >= 3]
    best_name, best = stable[0]
    mode, regularization_text, fraction_text = best_name.split("__")
    regularization = float(regularization_text[1:])
    fraction = float(fraction_text[1:])

    all_ids = np.concatenate([fold_data[fold]["ids"] for fold in FOLDS])
    raw_x = np.concatenate([fold_data[fold]["features"] for fold in FOLDS])
    raw_y = np.concatenate([fold_data[fold]["target"] for fold in FOLDS])
    aggregated_y = []
    all_x = None
    for column in range(raw_y.shape[1]):
        current_x, current_y = aggregate_training(all_ids, raw_x, raw_y[:, column])
        if all_x is None:
            all_x = current_x
        aggregated_y.append(current_y)
    final_model = fit_multi_ridge(
        all_x, np.column_stack(aggregated_y), mode, regularization
    )

    def save_model(source: Path, output: Path, model) -> None:
        payload = dict(np.load(source))
        coefficient, mean, std = model
        payload.update(
            sample_action_gate_coefficient=coefficient.astype(np.float32),
            sample_action_gate_mean=mean.astype(np.float32),
            sample_action_gate_std=std.astype(np.float32),
            sample_action_gate_fraction=np.array(fraction),
            sample_action_gate_actions=np.asarray(ACTIONS[1:]),
            sample_action_gate_target=np.array("joint_pas_pdp_optimal_nmse"),
            sample_action_gate_feature_mode=np.array(mode),
            sample_action_gate_regularization=np.array(regularization),
        )
        np.savez_compressed(output, **payload)

    save_model(
        Path("artifacts/v39_gated_spectral_s010.npz"),
        Path("artifacts/v41_four_action_sample_gate.npz"),
        final_model,
    )
    for fold in FOLDS:
        model = fitted[(mode, regularization)][fold]
        save_model(
            Path(f"artifacts/v39_gated_spectral_s010_split{fold}.npz"),
            Path(f"artifacts/v41_four_action_sample_gate_split{fold}.npz"),
            model,
        )
        action = choose_actions(
            predictions[(mode, regularization)][fold], fraction
        )
        payload = mixed_payload(fold_data[fold]["stats"], action)
        payload["sample_spectral_action_value"] = predictions[
            (mode, regularization)
        ][fold]
        np.savez_compressed(
            f"artifacts/v41_four_action_full_stats_split{fold}.npz", **payload
        )

    oracle_rows = []
    for fold in FOLDS:
        target = fold_data[fold]["target"]
        bank = np.column_stack((np.zeros(len(target)), target))
        oracle_action = np.argmax(bank, axis=1)
        oracle_rows.append(
            {
                "fold": fold,
                **action_metrics(fold_data[fold]["stats"], oracle_action),
            }
        )
    output = {
        "feature_width": int(all_x.shape[1]),
        "v40_score": float(np.mean(list(v40_fold_score.values()))),
        "oracle_score": float(np.mean([row["score"] for row in oracle_rows])),
        "oracle_folds": oracle_rows,
        "best": {"name": best_name, **best},
        "top": [[name, value] for name, value in top[:30]],
    }
    Path("artifacts/v41_four_action_sample_gate_search.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: output[k] for k in output if k != "top"}, indent=2))
    print(json.dumps({"top": output["top"][:10]}, indent=2))


if __name__ == "__main__":
    main()
