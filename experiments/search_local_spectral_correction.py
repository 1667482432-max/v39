from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from physical_ai.local_calibration import fit_metric_embedding
from physical_ai.spatial import ADVANCED_ENERGY_METRIC, ADVANCED_MAP_METRIC, metric_embeddings


FOLDS = ("101", "202", "20260804", "303", "404")
METRICS = ("xy_ctx-patch_s4", ADVANCED_MAP_METRIC, ADVANCED_ENERGY_METRIC)


def grouped(values: np.ndarray, kind: str) -> np.ndarray:
    if kind == "pas":
        return values[:, :1024].reshape(-1, 256, 4).transpose(0, 2, 1)
    return values[:, 1024:].reshape(-1, 2, 4, 192).reshape(-1, 8, 192)


def unit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, 1e-30), norm


def cosine(prediction: np.ndarray, target_unit: np.ndarray) -> float:
    prediction_unit, _ = unit(np.maximum(prediction, 0.0))
    return float(np.mean(np.sum(prediction_unit * target_unit, axis=-1)))


def aggregate(
    global_index: np.ndarray,
    values: np.ndarray,
    embedding_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _, first, inverse, count = np.unique(
        global_index, return_index=True, return_inverse=True, return_counts=True
    )
    output = np.zeros((len(count), *values.shape[1:]), dtype=np.float64)
    np.add.at(output, inverse, values)
    output /= count[(slice(None),) + (None,) * (values.ndim - 1)]
    return embedding_rows[first], output.astype(np.float32)


def search_kind(
    kind: str,
    stats: dict[str, dict[str, np.ndarray]],
    offsets: dict[str, np.ndarray],
    embeddings: dict[str, np.ndarray],
) -> tuple[dict[str, object], dict[str, list[dict[str, float | str]]]]:
    cache = {}
    for heldout in FOLDS:
        heldout_ids = set(stats[heldout]["global_index"].tolist())
        train_ids = []
        train_residual = []
        train_target = []
        train_embedding_rows = []
        for fold in FOLDS:
            if fold == heldout:
                continue
            keep = np.array(
                [index not in heldout_ids for index in stats[fold]["global_index"]]
            )
            prediction_unit, _ = unit(grouped(stats[fold]["prediction"], kind))
            target_unit, _ = unit(grouped(stats[fold]["target"], kind))
            train_ids.append(stats[fold]["global_index"][keep])
            train_residual.append((target_unit - prediction_unit)[keep])
            train_target.append(target_unit[keep])
            train_embedding_rows.append(offsets[fold][keep])
        ids = np.concatenate(train_ids)
        embedding_rows = np.concatenate(train_embedding_rows)
        reference_rows, residual = aggregate(
            ids, np.concatenate(train_residual), embedding_rows
        )
        target_rows, local_target = aggregate(
            ids, np.concatenate(train_target), embedding_rows
        )
        if not np.array_equal(reference_rows, target_rows):
            raise ValueError("Residual and target reference orders differ")
        query_prediction, _ = unit(grouped(stats[heldout]["prediction"], kind))
        query_target, _ = unit(grouped(stats[heldout]["target"], kind))
        local = {}
        for metric in METRICS:
            distance, indices = cKDTree(embeddings[metric][reference_rows]).query(
                embeddings[metric][offsets[heldout]], k=24, workers=-1
            )
            local[metric] = (distance, indices)
        cache[heldout] = {
            "prediction": query_prediction,
            "target": query_target,
            "residual": residual,
            "local_target": local_target,
            "local": local,
        }

    results: dict[str, list[dict[str, float | str]]] = {}
    for metric in METRICS:
        for neighbors in (4, 8, 16, 24):
            for power in (1.0, 2.0):
                for softening in (1.0, 3.0, 6.0):
                    predictions = {}
                    targets = {}
                    for heldout in FOLDS:
                        item = cache[heldout]
                        distance, indices = item["local"][metric]
                        distance = distance[:, :neighbors]
                        indices = indices[:, :neighbors]
                        weight = (distance + softening) ** (-power)
                        weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-30)
                        predictions[heldout] = np.einsum(
                            "qk,qkgl->qgl",
                            weight,
                            item["residual"][indices],
                            optimize=True,
                        )
                        targets[heldout] = np.einsum(
                            "qk,qkgl->qgl",
                            weight,
                            item["local_target"][indices],
                            optimize=True,
                        )
                    for method in ("residual", "target"):
                        for strength in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
                            rows = []
                            for heldout in FOLDS:
                                base = cache[heldout]["prediction"]
                                if method == "residual":
                                    corrected = base + strength * predictions[heldout]
                                else:
                                    corrected = (
                                        (1.0 - strength) * base
                                        + strength * targets[heldout]
                                    )
                                rows.append(
                                    {
                                        "fold": heldout,
                                        "score": cosine(
                                            corrected, cache[heldout]["target"]
                                        ),
                                    }
                                )
                            key = (
                                f"{metric}__k{neighbors}__p{power:g}__e{softening:g}"
                                f"__{method}__s{strength:g}"
                            )
                            results[key] = rows
    summary = {
        key: {"score": float(np.mean([row["score"] for row in rows]))}
        for key, rows in results.items()
    }
    top = sorted(summary.items(), key=lambda item: item[1]["score"], reverse=True)
    best_key = top[0][0]
    return {
        "baseline": float(
            np.mean(
                [
                    cosine(cache[fold]["prediction"], cache[fold]["target"])
                    for fold in FOLDS
                ]
            )
        ),
        "best": {
            "name": best_key,
            **summary[best_key],
            "folds": results[best_key],
        },
        "top": top[:30],
    }, results


def main() -> None:
    stats = {
        fold: dict(np.load(f"artifacts/v39_spectral_stats_split{fold}.npz"))
        for fold in FOLDS
    }
    offsets = {}
    cursor = 0
    for fold in FOLDS:
        offsets[fold] = np.arange(cursor, cursor + len(stats[fold]["position"]))
        cursor += len(stats[fold]["position"])
    all_positions = np.concatenate([stats[fold]["position"] for fold in FOLDS])
    all_contexts = np.concatenate([stats[fold]["context"] for fold in FOLDS])
    embeddings = metric_embeddings(all_positions, all_contexts)
    output = {}
    for kind in ("pas", "pdp"):
        result, _ = search_kind(kind, stats, offsets, embeddings)
        output[kind] = result
        print(json.dumps({kind: result}, indent=2), flush=True)

    all_ids = np.concatenate([stats[fold]["global_index"] for fold in FOLDS])
    all_prediction = np.concatenate([stats[fold]["prediction"] for fold in FOLDS])
    all_target = np.concatenate([stats[fold]["target"] for fold in FOLDS])
    model_payload: dict[str, np.ndarray] = {}
    selected: dict[str, dict[str, object]] = {}
    for kind in ("pas", "pdp"):
        parts = output[kind]["best"]["name"].split("__")
        metric = parts[0]
        neighbors = int(parts[1][1:])
        power = float(parts[2][1:])
        softening = float(parts[3][1:])
        method = parts[4]
        strength = float(parts[5][1:])
        selected[kind] = {
            "metric": metric,
            "neighbors": neighbors,
            "power": power,
            "softening": softening,
            "method": method,
            "strength": strength,
        }
        prediction_unit, _ = unit(grouped(all_prediction, kind))
        target_unit, _ = unit(grouped(all_target, kind))
        source = target_unit - prediction_unit if method == "residual" else target_unit
        reference_rows, reference_value = aggregate(
            all_ids, source, np.arange(len(all_positions))
        )
        fitted_embedding, mean, std, multiplier = fit_metric_embedding(
            all_positions, all_contexts, metric
        )
        np.testing.assert_allclose(
            fitted_embedding, embeddings[metric], rtol=1e-12, atol=1e-12
        )
        model_payload.update(
            {
                f"{kind}_reference_value": reference_value,
                f"{kind}_reference_embedding": fitted_embedding[reference_rows],
                f"{kind}_metric": np.array(metric),
                f"{kind}_context_mean": mean,
                f"{kind}_context_std": std,
                f"{kind}_context_multiplier": multiplier,
                f"{kind}_neighbors": np.array(neighbors),
                f"{kind}_power": np.array(power),
                f"{kind}_softening": np.array(softening),
                f"{kind}_method": np.array(method),
                f"{kind}_strength": np.array(strength),
            }
        )
    np.savez_compressed("artifacts/v39_local_spectral_correction.npz", **model_payload)
    for heldout in FOLDS:
        heldout_ids = set(stats[heldout]["global_index"].tolist())
        fold_payload: dict[str, np.ndarray] = {}
        for kind in ("pas", "pdp"):
            config = selected[kind]
            source_rows = []
            source_ids = []
            embedding_rows = []
            for fold in FOLDS:
                if fold == heldout:
                    continue
                keep = np.array(
                    [
                        index not in heldout_ids
                        for index in stats[fold]["global_index"]
                    ]
                )
                prediction_unit, _ = unit(grouped(stats[fold]["prediction"], kind))
                target_unit, _ = unit(grouped(stats[fold]["target"], kind))
                source = (
                    target_unit - prediction_unit
                    if config["method"] == "residual"
                    else target_unit
                )
                source_rows.append(source[keep])
                source_ids.append(stats[fold]["global_index"][keep])
                embedding_rows.append(offsets[fold][keep])
            reference_rows, reference_value = aggregate(
                np.concatenate(source_ids),
                np.concatenate(source_rows),
                np.concatenate(embedding_rows),
            )
            metric = str(config["metric"])
            fitted_embedding, mean, std, multiplier = fit_metric_embedding(
                all_positions, all_contexts, metric
            )
            fold_payload.update(
                {
                    f"{kind}_reference_value": reference_value,
                    f"{kind}_reference_embedding": fitted_embedding[reference_rows],
                    f"{kind}_metric": np.array(metric),
                    f"{kind}_context_mean": mean,
                    f"{kind}_context_std": std,
                    f"{kind}_context_multiplier": multiplier,
                    f"{kind}_neighbors": np.array(config["neighbors"]),
                    f"{kind}_power": np.array(config["power"]),
                    f"{kind}_softening": np.array(config["softening"]),
                    f"{kind}_method": np.array(config["method"]),
                    f"{kind}_strength": np.array(config["strength"]),
                }
            )
        np.savez_compressed(
            f"artifacts/v39_local_spectral_correction_split{heldout}.npz",
            **fold_payload,
        )
    Path("artifacts/v39_local_spectral_search.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
