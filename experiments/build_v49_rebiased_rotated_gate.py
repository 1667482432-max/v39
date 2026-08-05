from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.search_four_action_sample_gate import FOLDS, mixed_payload
from experiments.search_nonlinear_four_action_gate import load_data
from experiments.search_v46_action_bias import choose_biased_actions


FRACTION = 0.95
BIAS = np.array([0.0, 0.00019, 0.000025], dtype=np.float64)


def update_model(source: Path, output: Path) -> None:
    payload = dict(np.load(source))
    payload.update(
        sample_action_physical_fraction=np.array(FRACTION),
        sample_action_physical_bias=BIAS,
        sample_action_physical_objective=np.array(
            "joint_pas_pdp_optimal_nmse_rotated_rebiased"
        ),
    )
    np.savez_compressed(output, **payload)


def main() -> None:
    data = load_data()
    for fold in FOLDS:
        source = Path(f"artifacts/v48_rotated_action_gate_split{fold}.npz")
        stats_source = np.load(
            f"artifacts/v48_rotated_action_full_stats_split{fold}.npz"
        )
        value = stats_source["sample_spectral_action_value"].astype(np.float64)
        action = choose_biased_actions(value, FRACTION, BIAS[1], BIAS[2])
        payload = mixed_payload(data[fold]["stats"], action)
        payload["sample_spectral_action_value"] = value
        np.savez_compressed(
            f"artifacts/v49_rebiased_rotated_full_stats_split{fold}.npz", **payload
        )
        update_model(
            source, Path(f"artifacts/v49_rebiased_rotated_gate_split{fold}.npz")
        )
    update_model(
        Path("artifacts/v48_rotated_action_gate.npz"),
        Path("artifacts/v49_rebiased_rotated_gate.npz"),
    )


if __name__ == "__main__":
    main()
