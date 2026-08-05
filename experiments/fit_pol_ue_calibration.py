from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_FOLDS = ("20260804", "101", "202", "303", "404")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit leakage-free polarization/UE calibration")
    parser.add_argument("--stats-pattern", default="artifacts/calibration_stats_split{fold}.npz")
    parser.add_argument("--folds", nargs="+", default=DEFAULT_FOLDS)
    parser.add_argument("--output", type=Path, default=Path("artifacts/pol_ue_calibration.json"))
    return parser.parse_args()


def pooled_scale(stats: dict[str, dict[str, np.ndarray]], folds: list[str]) -> np.ndarray:
    cross = sum(stats[fold]["cross_pol_ue"].sum(axis=0) for fold in folds)
    energy = sum(stats[fold]["pred_energy_pol_ue"].sum(axis=0) for fold in folds)
    return cross / np.maximum(energy, 1e-30)


def encode(scale: np.ndarray) -> dict[str, list[list[float]]]:
    return {"real": scale.real.tolist(), "imag": scale.imag.tolist()}


def main() -> None:
    args = parse_args()
    folds = list(args.folds)
    stats = {
        fold: dict(np.load(args.stats_pattern.format(fold=fold)))
        for fold in folds
    }
    loo = []
    for held_out in folds:
        training = [fold for fold in folds if fold != held_out]
        loo.append({"fold": held_out, **encode(pooled_scale(stats, training))})
    result = {
        "folds": folds,
        "group_order": "bs_polarization_then_ue_antenna",
        "diagnostic_beta": 0.14,
        "loo": loo,
        "final": encode(pooled_scale(stats, folds)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
