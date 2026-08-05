from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from experiments.search_four_action_sample_gate import FOLDS
from experiments.search_nonlinear_four_action_gate import aggregate_multi, load_data
from experiments.search_v46_action_bias import choose_biased_actions, metrics, prepare_banks
from experiments.search_v47_rotated_geometry import BOTH_BIAS, PDP_BIAS, transform


def main() -> None:
    data = load_data()
    banks = {fold: prepare_banks(data[fold]["stats"]) for fold in FOLDS}
    source = {}
    for heldout in FOLDS:
        heldout_ids = set(data[heldout]["ids"].tolist())
        ids, positions, targets = [], [], []
        for fold in FOLDS:
            if fold == heldout:
                continue
            keep = np.asarray([index not in heldout_ids for index in data[fold]["ids"]])
            ids.append(data[fold]["ids"][keep])
            positions.append(data[fold]["stats"]["none"]["position"][keep, :2])
            targets.append(data[fold]["target"][keep])
        position, target = aggregate_multi(
            np.concatenate(ids), np.concatenate(positions), np.concatenate(targets)
        )
        source[heldout] = {"position": position, "target": target}

    candidates = []
    neighbor_grid = (4, 5, 6, 7, 8, 10)
    for angle in np.arange(157.5, 172.6, 2.5):
        for y_scale in (0.4, 0.45, 0.5, 0.55, 0.6):
            distances, indices = {}, {}
            for fold in FOLDS:
                distances[fold], indices[fold] = cKDTree(
                    transform(source[fold]["position"], angle, y_scale)
                ).query(
                    transform(data[fold]["stats"]["none"]["position"], angle, y_scale),
                    k=max(neighbor_grid),
                    workers=-1,
                )
            for neighbors in neighbor_grid:
                for power in (5.0, 5.5, 6.0, 6.5, 7.0):
                    for softening in (0.1, 0.2, 0.3, 0.5, 0.7):
                        values = {}
                        for fold in FOLDS:
                            distance = distances[fold][:, :neighbors]
                            local = indices[fold][:, :neighbors]
                            weight = (distance + softening) ** (-power)
                            weight /= weight.sum(axis=1, keepdims=True)
                            values[fold] = np.einsum(
                                "qk,qka->qa",
                                weight,
                                source[fold]["target"][local],
                                optimize=True,
                            )
                        for fraction in (0.94, 0.95, 0.96, 0.97):
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
    Path("artifacts/v48_fine_rotated_geometry_search.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"best": output["best"], "top": output["top"][:10]}, indent=2))


if __name__ == "__main__":
    main()
