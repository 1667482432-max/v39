from __future__ import annotations

import json

import numpy as np


FOLDS = ("101", "202", "20260804", "303", "404")
BASE_PAS = (0.764449529647827, 0.780791307818145, 0.771951640360057, 0.745609395541251, 0.756040462702513)
BASE_PDP = (0.862799538373947, 0.860145515203476, 0.84170221708715, 0.846051997169852, 0.848091188240796)


def optimal_nmse(stats: dict[str, np.ndarray]) -> float:
    cross = stats["final_cross"].sum()
    prediction_energy = stats["final_pred_energy"].sum()
    target_energy = stats["target_energy"].sum()
    return float(1.0 - np.abs(cross) ** 2 / (prediction_energy * target_energy))


def main() -> None:
    rows = []
    for index, fold in enumerate(FOLDS):
        candidate = dict(np.load(f"artifacts/v39_full_stats_split{fold}.npz"))
        baseline = dict(np.load(f"artifacts/v37_scalar_stats_split{fold}.npz"))
        rows.append(
            {
                "fold": fold,
                "pas": float(candidate["final_pas"].mean()),
                "base_pas": BASE_PAS[index],
                "pdp": float(candidate["final_pdp"].mean()),
                "base_pdp": BASE_PDP[index],
                "optimal_nmse": optimal_nmse(candidate),
                "base_optimal_nmse": optimal_nmse(baseline),
            }
        )
    output = {
        "folds": rows,
        "mean": {
            key: float(np.mean([row[key] for row in rows]))
            for key in (
                "pas",
                "base_pas",
                "pdp",
                "base_pdp",
                "optimal_nmse",
                "base_optimal_nmse",
            )
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
