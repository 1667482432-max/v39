from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.search_four_action_sample_gate import FOLDS
from experiments.search_nonlinear_four_action_gate import load_data
from experiments.search_v46_action_bias import evaluate, prepare_banks


def main() -> None:
    data = load_data()
    values = {
        fold: np.load(
            f"artifacts/v48_rotated_action_full_stats_split{fold}.npz"
        )["sample_spectral_action_value"].astype(np.float64)
        for fold in FOLDS
    }
    banks = {fold: prepare_banks(data[fold]["stats"]) for fold in FOLDS}
    candidates = []
    for pdp_bias in np.linspace(0.0001, 0.0005, 21):
        for both_bias in np.linspace(-0.00008, 0.00032, 21):
            for fraction in np.linspace(0.90, 1.0, 21):
                result = evaluate(data, values, banks, fraction, pdp_bias, both_bias)
                candidates.append(
                    {"fraction": fraction, "bias": [0.0, pdp_bias, both_bias], **result}
                )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0]
    fine = []
    for pdp_delta in np.linspace(-0.00004, 0.00004, 17):
        for both_delta in np.linspace(-0.00004, 0.00004, 17):
            for fraction_delta in np.linspace(-0.02, 0.02, 17):
                fraction = float(np.clip(best["fraction"] + fraction_delta, 0.0, 1.0))
                pdp_bias = float(best["bias"][1] + pdp_delta)
                both_bias = float(best["bias"][2] + both_delta)
                result = evaluate(data, values, banks, fraction, pdp_bias, both_bias)
                fine.append(
                    {"fraction": fraction, "bias": [0.0, pdp_bias, both_bias], **result}
                )
    candidates.extend(fine)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    output = {"best": candidates[0], "top": candidates[:50]}
    Path("artifacts/v49_rotated_action_bias_search.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"best": output["best"], "top": output["top"][:10]}, indent=2))


if __name__ == "__main__":
    main()
