from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from experiments.search_local_spectral_correction import FOLDS, aggregate, grouped, unit
from physical_ai.spatial import ADVANCED_ENERGY_METRIC, ADVANCED_MAP_METRIC, metric_embeddings


CONFIGS = {
    "pas": {
        "metric": ADVANCED_MAP_METRIC,
        "neighbors": 4,
        "power": 2.0,
        "softening": 1.0,
    },
    "pdp": {
        "metric": ADVANCED_ENERGY_METRIC,
        "neighbors": 24,
        "power": 2.0,
        "softening": 1.0,
    },
}


def score(prediction: np.ndarray, target: np.ndarray) -> float:
    normalized, _ = unit(np.maximum(prediction, 0.0))
    return float(np.mean(np.sum(normalized * target, axis=-1)))


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
    positions = np.concatenate([stats[fold]["position"] for fold in FOLDS])
    contexts = np.concatenate([stats[fold]["context"] for fold in FOLDS])
    embeddings = metric_embeddings(positions, contexts)
    output = {}
    model_updates = {}
    for kind in ("pas", "pdp"):
        config = CONFIGS[kind]
        cache = {}
        pooled = {"agreement": [], "consensus": [], "delta": []}
        for heldout in FOLDS:
            heldout_ids = set(stats[heldout]["global_index"].tolist())
            source_ids = []
            source_target = []
            embedding_rows = []
            for fold in FOLDS:
                if fold == heldout:
                    continue
                keep = np.array(
                    [index not in heldout_ids for index in stats[fold]["global_index"]]
                )
                target, _ = unit(grouped(stats[fold]["target"], kind))
                source_ids.append(stats[fold]["global_index"][keep])
                source_target.append(target[keep])
                embedding_rows.append(offsets[fold][keep])
            reference_rows, reference_target = aggregate(
                np.concatenate(source_ids),
                np.concatenate(source_target),
                np.concatenate(embedding_rows),
            )
            distance, local = cKDTree(
                embeddings[config["metric"]][reference_rows]
            ).query(
                embeddings[config["metric"]][offsets[heldout]],
                k=config["neighbors"],
                workers=-1,
            )
            if config["neighbors"] == 1:
                distance, local = distance[:, None], local[:, None]
            weight = (distance + config["softening"]) ** (-config["power"])
            weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-30)
            local_target = np.einsum(
                "qk,qkgl->qgl", weight, reference_target[local], optimize=True
            )
            base, _ = unit(grouped(stats[heldout]["prediction"], kind))
            target, _ = unit(grouped(stats[heldout]["target"], kind))
            consensus = np.linalg.norm(local_target, axis=-1)
            local_unit = local_target / np.maximum(consensus[..., None], 1e-30)
            agreement = np.sum(base * local_unit, axis=-1)
            delta = np.linalg.norm(local_target - base, axis=-1)
            features = {
                "agreement": agreement,
                "consensus": consensus,
                "delta": delta,
            }
            for name, value in features.items():
                pooled[name].append(value.reshape(-1))
            cache[heldout] = {
                "base": base,
                "target": target,
                "local": local_target,
                "features": features,
            }
        thresholds = {
            name: np.unique(
                np.quantile(np.concatenate(values), np.linspace(0.1, 0.9, 9))
            )
            for name, values in pooled.items()
        }
        candidates = [("always", "all", 0.0, None, None, 0.0)]
        for name, values in thresholds.items():
            for threshold in values:
                candidates.append((name, "low", float(threshold), None, None, 0.0))
                candidates.append((name, "high", float(threshold), None, None, 0.0))
        for consensus_threshold in thresholds["consensus"]:
            for agreement_threshold in thresholds["agreement"]:
                candidates.append(
                    (
                        "consensus_agreement",
                        "high_low",
                        float(consensus_threshold),
                        "agreement",
                        "low",
                        float(agreement_threshold),
                    )
                )
        results = {}
        for candidate in candidates:
            name, direction, threshold, second_name, second_direction, second_threshold = candidate
            for strength in (0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15):
                rows = []
                applied = []
                for heldout in FOLDS:
                    item = cache[heldout]
                    if name == "always":
                        mask = np.ones(item["base"].shape[:2], dtype=bool)
                    elif name == "consensus_agreement":
                        mask = item["features"]["consensus"] >= threshold
                        mask &= item["features"][second_name] <= second_threshold
                    elif direction == "low":
                        mask = item["features"][name] <= threshold
                    else:
                        mask = item["features"][name] >= threshold
                    corrected = item["base"] + strength * mask[..., None] * (
                        item["local"] - item["base"]
                    )
                    rows.append(
                        {"fold": heldout, "score": score(corrected, item["target"])}
                    )
                    applied.append(float(mask.mean()))
                key = (
                    f"{name}__{direction}__t{threshold:.9g}"
                    f"__t2{second_threshold:.9g}__s{strength:g}"
                )
                results[key] = {
                    "score": float(np.mean([row["score"] for row in rows])),
                    "applied": float(np.mean(applied)),
                    "folds": rows,
                    "candidate": candidate,
                    "strength": strength,
                }
        top = sorted(results.items(), key=lambda item: item[1]["score"], reverse=True)
        best_key, best = top[0]
        oracle_rows = []
        for heldout in FOLDS:
            item = cache[heldout]
            base_score = np.sum(item["base"] * item["target"], axis=-1)
            corrected = item["base"] + 0.05 * (item["local"] - item["base"])
            corrected_unit, _ = unit(np.maximum(corrected, 0.0))
            corrected_score = np.sum(corrected_unit * item["target"], axis=-1)
            oracle_rows.append(float(np.mean(np.maximum(base_score, corrected_score))))
        output[kind] = {
            "baseline": float(
                np.mean(
                    [score(cache[fold]["base"], cache[fold]["target"]) for fold in FOLDS]
                )
            ),
            "oracle_binary_s005": float(np.mean(oracle_rows)),
            "best": {"name": best_key, **{k: v for k, v in best.items() if k != "candidate"}},
            "top": [
                [key, {k: v for k, v in value.items() if k not in ("candidate", "folds")}]
                for key, value in top[:30]
            ],
        }
        candidate = best["candidate"]
        model_updates[kind] = {
            "gate_name": np.array(candidate[0]),
            "gate_direction": np.array(candidate[1]),
            "gate_threshold": np.array(candidate[2]),
            "gate_second_name": np.array(candidate[3] or "none"),
            "gate_second_direction": np.array(candidate[4] or "none"),
            "gate_second_threshold": np.array(candidate[5]),
            "strength": np.array(best["strength"]),
        }
        print(json.dumps({kind: output[kind]}, indent=2), flush=True)

    artifact = dict(np.load("artifacts/v39_local_spectral_correction.npz"))
    for kind, updates in model_updates.items():
        artifact[f"{kind}_strength"] = updates["strength"]
        for name, value in updates.items():
            if name == "strength":
                continue
            artifact[f"{kind}_{name}"] = value
    np.savez_compressed("artifacts/v39_gated_spectral_correction.npz", **artifact)
    for fold in FOLDS:
        fold_artifact = dict(
            np.load(f"artifacts/v39_local_spectral_correction_split{fold}.npz")
        )
        for kind, updates in model_updates.items():
            fold_artifact[f"{kind}_strength"] = updates["strength"]
            for name, value in updates.items():
                if name == "strength":
                    continue
                fold_artifact[f"{kind}_{name}"] = value
        np.savez_compressed(
            f"artifacts/v39_gated_spectral_correction_split{fold}.npz",
            **fold_artifact,
        )
    Path("artifacts/v39_spectral_confidence_gate.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
