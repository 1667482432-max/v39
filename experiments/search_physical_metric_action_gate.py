from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from experiments.search_four_action_sample_gate import (
    FOLDS,
    action_metrics,
    choose_actions,
    fit_multi_ridge,
    predict_multi_ridge,
)
from experiments.search_nonlinear_four_action_gate import aggregate_multi, load_data
from physical_ai.spatial import metric_embeddings


def main() -> None:
    data = load_data()
    offsets = {}
    cursor = 0
    position_rows, context_rows = [], []
    for fold in FOLDS:
        stats = data[fold]["stats"]["none"]
        offsets[fold] = np.arange(cursor, cursor + len(stats["position"]))
        cursor += len(stats["position"])
        position_rows.append(stats["position"])
        context_rows.append(stats["context"])
    embeddings = metric_embeddings(
        np.concatenate(position_rows), np.concatenate(context_rows)
    )
    metrics = (
        "xy_y0.75",
        "xy_ctx-patch_s4",
        "xy_ctx-material-center-multiscale_s3",
        "xy_ctx-endpoint-far_s4",
    )

    cache = {}
    for heldout in FOLDS:
        heldout_ids = set(data[heldout]["ids"].tolist())
        ids, feature_rows, target_rows, embedding_rows = [], [], [], []
        for fold in FOLDS:
            if fold == heldout:
                continue
            keep = np.asarray([index not in heldout_ids for index in data[fold]["ids"]])
            ids.append(data[fold]["ids"][keep])
            feature_rows.append(data[fold]["features"][keep])
            target_rows.append(data[fold]["target"][keep])
            embedding_rows.append(offsets[fold][keep])
        train_x, train_y = aggregate_multi(
            np.concatenate(ids), np.concatenate(feature_rows), np.concatenate(target_rows)
        )
        linear_model = fit_multi_ridge(train_x, train_y, "all", 100.0)
        linear = predict_multi_ridge(data[heldout]["features"], *linear_model)
        unique_ids, inverse = np.unique(np.concatenate(ids), return_inverse=True)
        del unique_ids
        count = np.bincount(inverse).astype(np.float64)
        source_rows = np.concatenate(embedding_rows)
        metric_reference = {}
        for metric in metrics:
            values = embeddings[metric][source_rows]
            total = np.zeros((len(count), values.shape[1]), dtype=np.float64)
            np.add.at(total, inverse, values)
            metric_reference[metric] = total / count[:, None]
        cache[heldout] = {
            "target": train_y,
            "linear": linear,
            "reference": metric_reference,
        }

    results = {}
    fractions = tuple(np.linspace(0.1, 0.9, 17))
    neighbor_counts = (8, 16, 24, 32, 48, 64, 96, 128)
    for metric in metrics:
        distances, indices = {}, {}
        for fold in FOLDS:
            distance, local = cKDTree(cache[fold]["reference"][metric]).query(
                embeddings[metric][offsets[fold]], k=max(neighbor_counts), workers=-1
            )
            distances[fold], indices[fold] = distance, local
        for neighbors in neighbor_counts:
            for power in (0.0, 0.5, 1.0, 2.0, 4.0):
                for softening in (0.1, 1.0, 3.0, 6.0):
                    local_prediction = {}
                    for fold in FOLDS:
                        distance = distances[fold][:, :neighbors]
                        local = indices[fold][:, :neighbors]
                        if power == 0.0:
                            weight = np.ones_like(distance)
                        else:
                            weight = (distance + softening) ** (-power)
                        weight /= weight.sum(axis=1, keepdims=True)
                        local_prediction[fold] = np.einsum(
                            "qk,qka->qa",
                            weight,
                            cache[fold]["target"][local],
                            optimize=True,
                        )
                    for blend in (0.25, 0.5, 0.75, 1.0):
                        prediction = {
                            fold: (1.0 - blend) * cache[fold]["linear"]
                            + blend * local_prediction[fold]
                            for fold in FOLDS
                        }
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
                                f"{metric}__k{neighbors}__p{power:g}__e{softening:g}"
                                f"__b{blend:g}__f{fraction:g}"
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
        "top": [[name, value] for name, value in top[:30]],
    }
    Path("artifacts/v44_physical_metric_action_search.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"best": output["best"], "top": output["top"][:10]}, indent=2))


if __name__ == "__main__":
    main()
