from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.build_v45_fine_xy_action_gate import (
    NEIGHBORS,
    POWER,
    SOFTENING,
    Y_SCALE,
    embedding,
    predict,
)
from experiments.search_four_action_sample_gate import FOLDS, mixed_payload
from experiments.search_nonlinear_four_action_gate import aggregate_multi, load_data
from experiments.search_v46_action_bias import choose_biased_actions


FRACTION = 0.97
BIAS = np.array([0.0, 0.00028, 0.00012], dtype=np.float64)


def save_model(
    source: Path, output: Path, reference: np.ndarray, target: np.ndarray
) -> None:
    payload = dict(np.load(source))
    payload.update(
        sample_action_physical_reference=reference,
        sample_action_physical_target=target.astype(np.float32),
        sample_action_physical_metric=np.array(f"xy_y{Y_SCALE:g}"),
        sample_action_physical_neighbors=np.array(NEIGHBORS),
        sample_action_physical_power=np.array(POWER),
        sample_action_physical_softening=np.array(SOFTENING),
        sample_action_physical_fraction=np.array(FRACTION),
        sample_action_physical_bias=BIAS,
        sample_action_physical_objective=np.array(
            "joint_pas_pdp_optimal_nmse_biased"
        ),
    )
    np.savez_compressed(output, **payload)


def main() -> None:
    data = load_data()
    for heldout in FOLDS:
        heldout_ids = set(data[heldout]["ids"].tolist())
        ids, positions, targets = [], [], []
        for fold in FOLDS:
            if fold == heldout:
                continue
            keep = np.asarray([index not in heldout_ids for index in data[fold]["ids"]])
            ids.append(data[fold]["ids"][keep])
            positions.append(embedding(data[fold]["stats"]["none"])[keep])
            targets.append(data[fold]["target"][keep])
        reference, target = aggregate_multi(
            np.concatenate(ids), np.concatenate(positions), np.concatenate(targets)
        )
        value = predict(reference, target, embedding(data[heldout]["stats"]["none"]))
        action = choose_biased_actions(value, FRACTION, BIAS[1], BIAS[2])
        payload = mixed_payload(data[heldout]["stats"], action)
        payload["sample_spectral_action_value"] = value
        np.savez_compressed(
            f"artifacts/v46_biased_action_full_stats_split{heldout}.npz", **payload
        )
        save_model(
            Path(f"artifacts/v39_gated_spectral_s010_split{heldout}.npz"),
            Path(f"artifacts/v46_biased_action_gate_split{heldout}.npz"),
            reference,
            target,
        )

    all_ids = np.concatenate([data[fold]["ids"] for fold in FOLDS])
    reference, target = aggregate_multi(
        all_ids,
        np.concatenate([embedding(data[fold]["stats"]["none"]) for fold in FOLDS]),
        np.concatenate([data[fold]["target"] for fold in FOLDS]),
    )
    save_model(
        Path("artifacts/v39_gated_spectral_s010.npz"),
        Path("artifacts/v46_biased_action_gate.npz"),
        reference,
        target,
    )


if __name__ == "__main__":
    main()
