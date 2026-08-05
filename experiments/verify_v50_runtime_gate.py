from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.search_four_action_sample_gate import FOLDS
from physical_ai.spectral_calibration import LocalSpectralCorrection


def main() -> None:
    results = []
    for fold in FOLDS:
        spectral = np.load(f"artifacts/v39_spectral_stats_split{fold}.npz")
        expected_stats = np.load(
            f"artifacts/v50_geometry_ensemble_full_stats_split{fold}.npz"
        )
        correction = LocalSpectralCorrection.load(
            Path(f"artifacts/v50_geometry_ensemble_gate_split{fold}.npz")
        )
        base_correction = LocalSpectralCorrection.load(
            Path(f"artifacts/v39_gated_spectral_s010_split{fold}.npz")
        )
        positions = spectral["position"]
        contexts = spectral["context"]
        compact = torch.from_numpy(spectral["prediction"].astype(np.float32))

        primary = correction._physical_action_value(
            positions,
            correction.sample_action_physical_reference,
            correction.sample_action_physical_target,
            correction.sample_action_physical_metric,
            correction.sample_action_physical_transform,
            correction.sample_action_physical_neighbors,
            correction.sample_action_physical_power,
            correction.sample_action_physical_softening,
        )
        secondary = correction._physical_action_value(
            positions,
            correction.sample_action_physical_secondary_reference,
            correction.sample_action_physical_secondary_target,
            correction.sample_action_physical_secondary_metric,
            correction.sample_action_physical_secondary_transform,
            correction.sample_action_physical_secondary_neighbors,
            correction.sample_action_physical_secondary_power,
            correction.sample_action_physical_secondary_softening,
        )
        value = (
            correction.sample_action_physical_ensemble_weight * primary
            + (1.0 - correction.sample_action_physical_ensemble_weight) * secondary
            + correction.sample_action_physical_bias
        )
        value_tensor = torch.from_numpy(value)
        benefit, action = torch.max(value_tensor, dim=1)
        count = int(round(correction.sample_action_physical_fraction * len(value)))
        selected = torch.zeros(len(value), dtype=torch.bool)
        if count:
            selected[torch.topk(benefit, count, sorted=False).indices] = True
        action = torch.where(selected, action + 1, 0).numpy()
        expected_action = expected_stats["sample_spectral_action"]
        if not np.array_equal(action, expected_action):
            raise AssertionError(f"Fold {fold} runtime action mismatch")

        corrected = correction.apply(compact, positions, contexts)
        both = base_correction.apply(compact, positions, contexts)
        expected = compact.clone()
        pas_selected = torch.from_numpy((action == 1) | (action == 3))
        pdp_selected = torch.from_numpy((action == 2) | (action == 3))
        expected[pas_selected, :1024] = both[pas_selected, :1024]
        expected[pdp_selected, 1024:] = both[pdp_selected, 1024:]
        maximum_error = float(torch.max(torch.abs(corrected - expected)).item())
        if maximum_error > 1e-6:
            raise AssertionError(
                f"Fold {fold} corrected compact mismatch: {maximum_error}"
            )
        results.append(
            {
                "fold": fold,
                "samples": len(action),
                "action_match": True,
                "maximum_compact_error": maximum_error,
            }
        )
    output = {"status": "passed", "folds": results}
    Path("artifacts/v50_runtime_gate_verification.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
