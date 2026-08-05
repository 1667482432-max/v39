from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from experiments.search_four_action_sample_gate import (
    FOLDS,
    choose_actions,
    mixed_payload,
)
from experiments.search_nonlinear_four_action_gate import aggregate_multi, load_data


NEIGHBORS = 48
POWER = 4.0
SOFTENING = 0.1
FRACTION = 0.75


def embedding(stats: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(stats["position"], dtype=np.float64)[:, :2] * np.array(
        [1.0, 0.75]
    )


def predict(reference: np.ndarray, target: np.ndarray, query: np.ndarray) -> np.ndarray:
    distance, local = cKDTree(reference).query(query, k=NEIGHBORS, workers=-1)
    weight = (distance + SOFTENING) ** (-POWER)
    weight /= weight.sum(axis=1, keepdims=True)
    return np.einsum("qk,qka->qa", weight, target[local], optimize=True)


def save_model(
    source: Path,
    output: Path,
    reference: np.ndarray,
    target: np.ndarray,
) -> None:
    payload = dict(np.load(source))
    payload.update(
        sample_action_physical_reference=reference,
        sample_action_physical_target=target.astype(np.float32),
        sample_action_physical_metric=np.array("xy_y0.75"),
        sample_action_physical_neighbors=np.array(NEIGHBORS),
        sample_action_physical_power=np.array(POWER),
        sample_action_physical_softening=np.array(SOFTENING),
        sample_action_physical_fraction=np.array(FRACTION),
        sample_action_physical_objective=np.array("joint_pas_pdp_optimal_nmse"),
    )
    np.savez_compressed(output, **payload)


def main() -> None:
    data = load_data()
    for heldout in FOLDS:
        heldout_ids = set(data[heldout]["ids"].tolist())
        ids, embedding_rows, target_rows = [], [], []
        for fold in FOLDS:
            if fold == heldout:
                continue
            keep = np.asarray([index not in heldout_ids for index in data[fold]["ids"]])
            ids.append(data[fold]["ids"][keep])
            embedding_rows.append(embedding(data[fold]["stats"]["none"])[keep])
            target_rows.append(data[fold]["target"][keep])
        reference, target = aggregate_multi(
            np.concatenate(ids),
            np.concatenate(embedding_rows),
            np.concatenate(target_rows),
        )
        value = predict(
            reference,
            target,
            embedding(data[heldout]["stats"]["none"]),
        )
        action = choose_actions(value, FRACTION)
        payload = mixed_payload(data[heldout]["stats"], action)
        payload["sample_spectral_action_value"] = value
        np.savez_compressed(
            f"artifacts/v44_physical_action_full_stats_split{heldout}.npz", **payload
        )
        save_model(
            Path(f"artifacts/v39_gated_spectral_s010_split{heldout}.npz"),
            Path(f"artifacts/v44_physical_action_gate_split{heldout}.npz"),
            reference,
            target,
        )

    all_ids = np.concatenate([data[fold]["ids"] for fold in FOLDS])
    reference, target = aggregate_multi(
        all_ids,
        np.concatenate(
            [embedding(data[fold]["stats"]["none"]) for fold in FOLDS]
        ),
        np.concatenate([data[fold]["target"] for fold in FOLDS]),
    )
    save_model(
        Path("artifacts/v39_gated_spectral_s010.npz"),
        Path("artifacts/v44_physical_action_gate.npz"),
        reference,
        target,
    )


if __name__ == "__main__":
    main()
