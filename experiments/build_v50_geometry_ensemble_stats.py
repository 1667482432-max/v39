from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.search_four_action_sample_gate import FOLDS, mixed_payload
from experiments.search_nonlinear_four_action_gate import load_data
from experiments.search_v46_action_bias import choose_biased_actions


ROTATED_WEIGHT = 0.37
FRACTION = 0.93
PDP_BIAS = 0.00012
BOTH_BIAS = 0.000045


def save_model(rotated: Path, original: Path, output: Path) -> None:
    primary = dict(np.load(rotated))
    secondary = np.load(original)
    primary.update(
        sample_action_physical_fraction=np.array(FRACTION),
        sample_action_physical_bias=np.array(
            [0.0, PDP_BIAS, BOTH_BIAS], dtype=np.float64
        ),
        sample_action_physical_secondary_reference=secondary[
            "sample_action_physical_reference"
        ],
        sample_action_physical_secondary_target=secondary[
            "sample_action_physical_target"
        ],
        sample_action_physical_secondary_metric=secondary[
            "sample_action_physical_metric"
        ],
        sample_action_physical_secondary_neighbors=secondary[
            "sample_action_physical_neighbors"
        ],
        sample_action_physical_secondary_power=secondary[
            "sample_action_physical_power"
        ],
        sample_action_physical_secondary_softening=secondary[
            "sample_action_physical_softening"
        ],
        sample_action_physical_ensemble_weight=np.array(ROTATED_WEIGHT),
        sample_action_physical_objective=np.array(
            "joint_pas_pdp_optimal_nmse_geometry_ensemble"
        ),
    )
    if "sample_action_physical_transform" in secondary:
        primary["sample_action_physical_secondary_transform"] = secondary[
            "sample_action_physical_transform"
        ]
    np.savez_compressed(output, **primary)


def main() -> None:
    data = load_data()
    for fold in FOLDS:
        original = np.load(
            f"artifacts/v45_fine_xy_action_full_stats_split{fold}.npz"
        )["sample_spectral_action_value"].astype(np.float64)
        rotated = np.load(
            f"artifacts/v48_rotated_action_full_stats_split{fold}.npz"
        )["sample_spectral_action_value"].astype(np.float64)
        value = (1.0 - ROTATED_WEIGHT) * original + ROTATED_WEIGHT * rotated
        action = choose_biased_actions(value, FRACTION, PDP_BIAS, BOTH_BIAS)
        payload = mixed_payload(data[fold]["stats"], action)
        payload["sample_spectral_action_value"] = value
        np.savez_compressed(
            f"artifacts/v50_geometry_ensemble_full_stats_split{fold}.npz", **payload
        )
        save_model(
            Path(f"artifacts/v48_rotated_action_gate_split{fold}.npz"),
            Path(f"artifacts/v45_fine_xy_action_gate_split{fold}.npz"),
            Path(f"artifacts/v50_geometry_ensemble_gate_split{fold}.npz"),
        )
    save_model(
        Path("artifacts/v48_rotated_action_gate.npz"),
        Path("artifacts/v45_fine_xy_action_gate.npz"),
        Path("artifacts/v50_geometry_ensemble_gate.npz"),
    )


if __name__ == "__main__":
    main()
