from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from experiments.search_four_action_sample_gate import (
    FOLDS,
    choose_actions,
    mixed_payload,
)
from experiments.search_nonlinear_four_action_gate import (
    FEATURE_MODES,
    aggregate_multi,
    load_data,
)


NEIGHBORS = 96
POWER = 1.0
SOFTENING = 3.0
FRACTION = 0.85
COLUMNS = FEATURE_MODES["condition"]


def predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference = (train_x[:, COLUMNS] - mean[COLUMNS]) / std[COLUMNS]
    query = (query_x[:, COLUMNS] - mean[COLUMNS]) / std[COLUMNS]
    distance, local = cKDTree(reference).query(query, k=NEIGHBORS, workers=-1)
    weight = (distance + SOFTENING) ** (-POWER)
    weight /= weight.sum(axis=1, keepdims=True)
    value = np.einsum("qk,qka->qa", weight, train_y[local], optimize=True)
    return reference, value


def save_model(
    source: Path,
    output: Path,
    train_x: np.ndarray,
    train_y: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    reference = (train_x[:, COLUMNS] - mean[COLUMNS]) / std[COLUMNS]
    payload = dict(np.load(source))
    payload.update(
        sample_action_knn_reference=reference.astype(np.float32),
        sample_action_knn_target=train_y.astype(np.float32),
        sample_action_knn_columns=COLUMNS,
        sample_action_knn_mean=mean.astype(np.float32),
        sample_action_knn_std=std.astype(np.float32),
        sample_action_knn_neighbors=np.array(NEIGHBORS),
        sample_action_knn_power=np.array(POWER),
        sample_action_knn_softening=np.array(SOFTENING),
        sample_action_knn_fraction=np.array(FRACTION),
        sample_action_knn_objective=np.array("joint_pas_pdp_optimal_nmse"),
    )
    np.savez_compressed(output, **payload)


def main() -> None:
    data = load_data()
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
        _, value = predict(
            train_x, train_y, data[heldout]["features"], mean, std
        )
        action = choose_actions(value, FRACTION)
        payload = mixed_payload(data[heldout]["stats"], action)
        payload["sample_spectral_action_value"] = value
        np.savez_compressed(
            f"artifacts/v42_knn_action_full_stats_split{heldout}.npz", **payload
        )
        save_model(
            Path(f"artifacts/v39_gated_spectral_s010_split{heldout}.npz"),
            Path(f"artifacts/v42_knn_action_gate_split{heldout}.npz"),
            train_x,
            train_y,
            mean,
            std,
        )

    all_ids = np.concatenate([data[fold]["ids"] for fold in FOLDS])
    all_x, all_y = aggregate_multi(
        all_ids,
        np.concatenate([data[fold]["features"] for fold in FOLDS]),
        np.concatenate([data[fold]["target"] for fold in FOLDS]),
    )
    mean = all_x.mean(axis=0)
    std = np.maximum(all_x.std(axis=0), 1e-6)
    save_model(
        Path("artifacts/v39_gated_spectral_s010.npz"),
        Path("artifacts/v42_knn_action_gate.npz"),
        all_x,
        all_y,
        mean,
        std,
    )


if __name__ == "__main__":
    main()
