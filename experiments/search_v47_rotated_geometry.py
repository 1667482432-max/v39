from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from experiments.search_four_action_sample_gate import FOLDS
from experiments.search_nonlinear_four_action_gate import aggregate_multi, load_data
from experiments.search_v46_action_bias import (
    choose_biased_actions,
    metrics,
    prepare_banks,
)


PDP_BIAS = 0.00028
BOTH_BIAS = 0.00012


def transform(position: np.ndarray, angle: float, y_scale: float) -> np.ndarray:
    radians = np.deg2rad(angle)
    rotation = np.array(
        [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]
    )
    return np.asarray(position, dtype=np.float64)[:, :2] @ rotation * np.array(
        [1.0, y_scale]
    )


def main() -> None:
    data = load_data()
    banks = {fold: prepare_banks(data[fold]["stats"]) for fold in FOLDS}
    source = {}
    for heldout in FOLDS:
        heldout_ids = set(data[heldout]["ids"].tolist())
        ids, position_rows, target_rows = [], [], []
        for fold in FOLDS:
            if fold == heldout:
                continue
            keep = np.asarray([index not in heldout_ids for index in data[fold]["ids"]])
            ids.append(data[fold]["ids"][keep])
            position_rows.append(data[fold]["stats"]["none"]["position"][keep, :2])
            target_rows.append(data[fold]["target"][keep])
        position, target = aggregate_multi(
            np.concatenate(ids), np.concatenate(position_rows), np.concatenate(target_rows)
        )
        source[heldout] = {"position": position, "target": target}

    candidates = []
    neighbors_grid = (6, 8, 10, 12, 16, 24, 32)
    for angle in np.arange(0.0, 180.0, 15.0):
        for y_scale in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            distance_cache, local_cache = {}, {}
            for fold in FOLDS:
                distance, local = cKDTree(
                    transform(source[fold]["position"], angle, y_scale)
                ).query(
                    transform(data[fold]["stats"]["none"]["position"], angle, y_scale),
                    k=max(neighbors_grid),
                    workers=-1,
                )
                distance_cache[fold], local_cache[fold] = distance, local
            for neighbors in neighbors_grid:
                for power in (4.0, 5.0, 6.0, 7.0, 8.0, 10.0):
                    for softening in (0.03, 0.1, 0.3, 1.0):
                        values = {}
                        for fold in FOLDS:
                            distance = distance_cache[fold][:, :neighbors]
                            local = local_cache[fold][:, :neighbors]
                            weight = (distance + softening) ** (-power)
                            weight /= weight.sum(axis=1, keepdims=True)
                            values[fold] = np.einsum(
                                "qk,qka->qa",
                                weight,
                                source[fold]["target"][local],
                                optimize=True,
                            )
                        for fraction in (0.94, 0.95, 0.96, 0.97, 0.98, 1.0):
                            rows = []
                            for fold in FOLDS:
                                action = choose_biased_actions(
                                    values[fold], fraction, PDP_BIAS, BOTH_BIAS
                                )
                                rows.append({"fold": fold, **metrics(banks[fold], action)})
                            candidates.append(
                                {
                                    "angle": float(angle),
                                    "y_scale": y_scale,
                                    "neighbors": neighbors,
                                    "power": power,
                                    "softening": softening,
                                    "fraction": fraction,
                                    "bias": [0.0, PDP_BIAS, BOTH_BIAS],
                                    "score": float(np.mean([row["score"] for row in rows])),
                                    "pas": float(np.mean([row["pas"] for row in rows])),
                                    "pdp": float(np.mean([row["pdp"] for row in rows])),
                                    "nmse": float(np.mean([row["nmse"] for row in rows])),
                                    "folds": rows,
                                }
                            )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    output = {"best": candidates[0], "top": candidates[:50]}
    Path("artifacts/v47_rotated_geometry_search.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"best": output["best"], "top": output["top"][:10]}, indent=2))


if __name__ == "__main__":
    main()
