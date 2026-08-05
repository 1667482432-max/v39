from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from physical_ai.spectral_calibration import LocalSpectralCorrection


FOLDS = ("101", "202", "20260804", "303", "404")
CONDITION_COLUMNS = 41


def optimal_nmse(stats: dict[str, np.ndarray], mask: np.ndarray | None = None) -> float:
    if mask is None:
        cross = stats["final_cross"].sum()
        prediction_energy = stats["final_pred_energy"].sum()
    else:
        base, corrected = stats["base"], stats["corrected"]
        cross = np.where(mask, corrected["final_cross"], base["final_cross"]).sum()
        prediction_energy = np.where(
            mask, corrected["final_pred_energy"], base["final_pred_energy"]
        ).sum()
    target_energy = stats["target_energy"].sum()
    return float(
        1.0 - np.abs(cross) ** 2 / max(prediction_energy * target_energy, 1e-30)
    )


def combined_metrics(item: dict[str, object], mask: np.ndarray) -> dict[str, float]:
    base = item["base"]
    corrected = item["corrected"]
    pas = float(np.where(mask, corrected["final_pas"], base["final_pas"]).mean())
    pdp = float(np.where(mask, corrected["final_pdp"], base["final_pdp"]).mean())
    nmse = optimal_nmse(item, mask)
    return {
        "pas": pas,
        "pdp": pdp,
        "nmse": nmse,
        "score": 0.4 * pas + 0.4 * pdp + 0.2 / (1.0 + nmse),
        "fraction": float(mask.mean()),
    }


def sample_joint_delta(
    base: dict[str, np.ndarray], corrected: dict[str, np.ndarray]
) -> np.ndarray:
    def sample_nmse(stats: dict[str, np.ndarray]) -> np.ndarray:
        return 1.0 - np.abs(stats["final_cross"]) ** 2 / np.maximum(
            stats["final_pred_energy"] * stats["target_energy"], 1e-30
        )

    base_nmse = sample_nmse(base)
    corrected_nmse = sample_nmse(corrected)
    return (
        0.4 * (corrected["final_pas"] - base["final_pas"])
        + 0.4 * (corrected["final_pdp"] - base["final_pdp"])
        + 0.2
        * (1.0 / (1.0 + corrected_nmse) - 1.0 / (1.0 + base_nmse))
    )


def aggregate_training(
    ids: np.ndarray, features: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    unique, inverse = np.unique(ids, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    feature_sum = np.zeros((len(unique), features.shape[1]), dtype=np.float64)
    target_sum = np.zeros(len(unique), dtype=np.float64)
    np.add.at(feature_sum, inverse, features)
    np.add.at(target_sum, inverse, target)
    return feature_sum / count[:, None], target_sum / count


def selected_columns(mode: str, width: int) -> np.ndarray:
    if mode == "all":
        return np.arange(width)
    spectral = np.arange(CONDITION_COLUMNS, width)
    if mode == "spectral":
        return spectral
    if mode == "basic_spectral":
        return np.concatenate((np.arange(9), spectral))
    raise ValueError(mode)


def fit_ridge(
    features: np.ndarray, target: np.ndarray, mode: str, regularization: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = np.maximum(features.std(axis=0), 1e-6)
    columns = selected_columns(mode, features.shape[1])
    x = (features[:, columns] - mean[columns]) / std[columns]
    centered = target - target.mean()
    system = x.T @ x + regularization * np.eye(len(columns))
    coefficient = np.linalg.solve(system, x.T @ centered)
    full = np.zeros(features.shape[1] + 1, dtype=np.float64)
    full[columns] = coefficient
    full[-1] = target.mean()
    return full, mean, std


def predict_ridge(
    features: np.ndarray,
    coefficient: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return ((features - mean) / np.maximum(std, 1e-6)) @ coefficient[:-1] + coefficient[-1]


def top_fraction_mask(value: np.ndarray, fraction: float) -> np.ndarray:
    count = min(max(int(round(fraction * len(value))), 0), len(value))
    mask = np.zeros(len(value), dtype=bool)
    if count:
        selected = np.argpartition(value, len(value) - count)[-count:]
        mask[selected] = True
    return mask


def main() -> None:
    fold_data: dict[str, dict[str, object]] = {}
    for fold in FOLDS:
        base = dict(np.load(f"artifacts/v40_nospec_full_stats_split{fold}.npz"))
        corrected = dict(np.load(f"artifacts/v39_s010_full_stats_split{fold}.npz"))
        spectral = dict(np.load(f"artifacts/v39_spectral_stats_split{fold}.npz"))
        np.testing.assert_array_equal(base["global_index"], corrected["global_index"])
        np.testing.assert_array_equal(base["global_index"], spectral["global_index"])
        correction = LocalSpectralCorrection.load(
            Path(f"artifacts/v39_gated_spectral_s010_split{fold}.npz")
        )
        features = correction.sample_features(
            torch.from_numpy(np.asarray(spectral["prediction"], dtype=np.float32)),
            spectral["position"],
            spectral["context"],
        ).astype(np.float64)
        fold_data[fold] = {
            "base": base,
            "corrected": corrected,
            "target_energy": base["target_energy"],
            "ids": base["global_index"],
            "features": features,
            "target": sample_joint_delta(base, corrected),
        }

    modes = ("spectral", "basic_spectral", "all")
    regularizations = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    fractions = tuple(np.linspace(0.0, 1.0, 21))
    results: dict[str, dict[str, object]] = {}
    fitted: dict[tuple[str, float], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    predictions: dict[tuple[str, float], dict[str, np.ndarray]] = {}
    for mode in modes:
        for regularization in regularizations:
            key = (mode, regularization)
            fitted[key] = {}
            predictions[key] = {}
            for heldout in FOLDS:
                heldout_ids = set(fold_data[heldout]["ids"].tolist())
                ids, feature_rows, target_rows = [], [], []
                for fold in FOLDS:
                    if fold == heldout:
                        continue
                    keep = np.asarray(
                        [index not in heldout_ids for index in fold_data[fold]["ids"]]
                    )
                    ids.append(fold_data[fold]["ids"][keep])
                    feature_rows.append(fold_data[fold]["features"][keep])
                    target_rows.append(fold_data[fold]["target"][keep])
                train_x, train_y = aggregate_training(
                    np.concatenate(ids),
                    np.concatenate(feature_rows),
                    np.concatenate(target_rows),
                )
                model = fit_ridge(train_x, train_y, mode, regularization)
                fitted[key][heldout] = model
                predictions[key][heldout] = predict_ridge(
                    fold_data[heldout]["features"], *model
                )
            for fraction in fractions:
                rows = []
                for heldout in FOLDS:
                    mask = top_fraction_mask(predictions[key][heldout], fraction)
                    rows.append(
                        {"fold": heldout, **combined_metrics(fold_data[heldout], mask)}
                    )
                name = f"{mode}__r{regularization:g}__f{fraction:g}"
                results[name] = {
                    "score": float(np.mean([row["score"] for row in rows])),
                    "pas": float(np.mean([row["pas"] for row in rows])),
                    "pdp": float(np.mean([row["pdp"] for row in rows])),
                    "nmse": float(np.mean([row["nmse"] for row in rows])),
                    "positive_folds": int(
                        sum(
                            row["score"]
                            > combined_metrics(
                                fold_data[row["fold"]],
                                np.zeros(len(fold_data[row["fold"]]["ids"]), dtype=bool),
                            )["score"]
                            for row in rows
                        )
                    ),
                    "folds": rows,
                }

    top = sorted(results.items(), key=lambda item: item[1]["score"], reverse=True)
    stable = [item for item in top if item[1]["positive_folds"] >= 3]
    best_name, best = stable[0]
    mode, regularization_text, fraction_text = best_name.split("__")
    regularization = float(regularization_text[1:])
    fraction = float(fraction_text[1:])

    all_ids = np.concatenate([fold_data[fold]["ids"] for fold in FOLDS])
    all_x, all_y = aggregate_training(
        all_ids,
        np.concatenate([fold_data[fold]["features"] for fold in FOLDS]),
        np.concatenate([fold_data[fold]["target"] for fold in FOLDS]),
    )
    final_model = fit_ridge(all_x, all_y, mode, regularization)

    def save_model(source: Path, output: Path, model) -> None:
        payload = dict(np.load(source))
        coefficient, mean, std = model
        payload.update(
            sample_gate_coefficient=coefficient.astype(np.float32),
            sample_gate_mean=mean.astype(np.float32),
            sample_gate_std=std.astype(np.float32),
            sample_gate_fraction=np.array(fraction),
            sample_gate_target=np.array("joint_pas_pdp_optimal_nmse"),
            sample_gate_feature_mode=np.array(mode),
            sample_gate_regularization=np.array(regularization),
        )
        np.savez_compressed(output, **payload)

    save_model(
        Path("artifacts/v39_gated_spectral_s010.npz"),
        Path("artifacts/v40_joint_sample_gate.npz"),
        final_model,
    )
    for fold in FOLDS:
        save_model(
            Path(f"artifacts/v39_gated_spectral_s010_split{fold}.npz"),
            Path(f"artifacts/v40_joint_sample_gate_split{fold}.npz"),
            fitted[(mode, regularization)][fold],
        )
        mask = top_fraction_mask(
            predictions[(mode, regularization)][fold], fraction
        )
        base = fold_data[fold]["base"]
        corrected = fold_data[fold]["corrected"]
        mixed = {}
        for name, base_value in base.items():
            corrected_value = corrected.get(name)
            if (
                corrected_value is not None
                and base_value.ndim > 0
                and corrected_value.shape == base_value.shape
                and len(base_value) == len(mask)
            ):
                reshape = (len(mask),) + (1,) * (base_value.ndim - 1)
                mixed[name] = np.where(mask.reshape(reshape), corrected_value, base_value)
            else:
                mixed[name] = base_value
        mixed["sample_spectral_gate"] = mask
        mixed["sample_spectral_gate_value"] = predictions[(mode, regularization)][fold]
        np.savez_compressed(
            f"artifacts/v40_joint_full_stats_split{fold}.npz", **mixed
        )

    oracle_rows = []
    baseline_rows = []
    corrected_rows = []
    for fold in FOLDS:
        oracle_rows.append(
            {
                "fold": fold,
                **combined_metrics(fold_data[fold], fold_data[fold]["target"] > 0.0),
            }
        )
        baseline_rows.append(
            combined_metrics(
                fold_data[fold], np.zeros(len(fold_data[fold]["ids"]), dtype=bool)
            )
        )
        corrected_rows.append(
            combined_metrics(
                fold_data[fold], np.ones(len(fold_data[fold]["ids"]), dtype=bool)
            )
        )
    output = {
        "feature_width": int(all_x.shape[1]),
        "baseline_score": float(np.mean([row["score"] for row in baseline_rows])),
        "fixed_corrected_score": float(
            np.mean([row["score"] for row in corrected_rows])
        ),
        "oracle_score": float(np.mean([row["score"] for row in oracle_rows])),
        "oracle_fraction": float(np.mean([row["fraction"] for row in oracle_rows])),
        "best": {"name": best_name, **best},
        "top": [[name, value] for name, value in top[:30]],
    }
    Path("artifacts/v40_joint_sample_gate_search.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: output[k] for k in output if k != "top"}, indent=2))
    print(json.dumps({"top": output["top"][:10]}, indent=2))


if __name__ == "__main__":
    main()
