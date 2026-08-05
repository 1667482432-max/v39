from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from experiments.search_four_action_sample_gate import FOLDS, action_metrics, choose_actions
from experiments.search_nonlinear_four_action_gate import aggregate_multi, load_data


def main() -> None:
    data = load_data()
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

    results = {}
    neighbor_counts = (8, 12, 16, 20, 24, 32, 40, 48, 64, 96)
    fractions = tuple(np.linspace(0.4, 0.95, 12))
    for y_scale in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        distance_cache, local_cache = {}, {}
        for fold in FOLDS:
            scale = np.array([1.0, y_scale])
            distance, local = cKDTree(source[fold]["position"] * scale).query(
                data[fold]["stats"]["none"]["position"][:, :2] * scale,
                k=max(neighbor_counts),
                workers=-1,
            )
            distance_cache[fold], local_cache[fold] = distance, local
        for neighbors in neighbor_counts:
            for power in (3.0, 4.0, 6.0, 8.0, 12.0):
                for softening in (0.0, 0.01, 0.03, 0.1, 0.3, 1.0):
                    prediction = {}
                    for fold in FOLDS:
                        distance = distance_cache[fold][:, :neighbors]
                        local = local_cache[fold][:, :neighbors]
                        weight = (distance + softening + 1e-9) ** (-power)
                        weight /= weight.sum(axis=1, keepdims=True)
                        prediction[fold] = np.einsum(
                            "qk,qka->qa",
                            weight,
                            source[fold]["target"][local],
                            optimize=True,
                        )
                    for fraction in fractions:
                        rows = []
                        for fold in FOLDS:
                            action = choose_actions(prediction[fold], fraction)
                            rows.append(
                                {
                                    "fold": fold,
                                    **action_metrics(data[fold]["stats"], action),
                                }
                            )
                        name = (
                            f"y{y_scale:g}__k{neighbors}__p{power:g}"
                            f"__e{softening:g}__f{fraction:g}"
                        )
                        results[name] = {
                            "score": float(np.mean([row["score"] for row in rows])),
                            "pas": float(np.mean([row["pas"] for row in rows])),
                            "pdp": float(np.mean([row["pdp"] for row in rows])),
                            "nmse": float(np.mean([row["nmse"] for row in rows])),
                            "folds": rows,
                        }
    top = sorted(results.items(), key=lambda item: item[1]["score"], reverse=True)
    output = {
        "best": {"name": top[0][0], **top[0][1]},
        "top": [[name, value] for name, value in top[:50]],
    }
    Path("artifacts/v45_fine_xy_action_search.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"best": output["best"], "top": output["top"][:15]}, indent=2))


if __name__ == "__main__":
    main()
