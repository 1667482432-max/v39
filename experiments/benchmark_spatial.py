from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from physical_ai.data import RoundData
from physical_ai.neighbors import (
    affine_reproduction_weights,
    delaunay_neighbors,
    distance_weights,
    nearest_neighbors,
)


def cos_sum(pred: np.ndarray, target: np.ndarray, axis: int) -> tuple[float, int]:
    dot = np.sum(pred * target, axis=axis, dtype=np.float64)
    pn = np.sqrt(np.sum(pred * pred, axis=axis, dtype=np.float64))
    tn = np.sqrt(np.sum(target * target, axis=axis, dtype=np.float64))
    value = dot / np.maximum(pn * tn, 1e-30)
    return float(value.sum()), value.size


def unit_mix(source: np.ndarray, weights: np.ndarray, axis: int) -> np.ndarray:
    norm = np.sqrt(np.sum(source * source, axis=axis, keepdims=True, dtype=np.float64))
    unit = source / np.maximum(norm, 1e-30)
    mixed = np.einsum("k,kmns->mns", weights, unit, optimize=True)
    return np.maximum(mixed, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare geometry-aware spectral interpolators")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output", type=Path, default=Path("artifacts/spatial_benchmark.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = RoundData(args.root)
    data.validate()
    positions = np.asarray(data.train_positions)
    channels = data.train_channels
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(positions))
    val_idx = np.sort(permutation[: args.validation_size])
    train_idx = np.sort(permutation[args.validation_size :])
    query, reference = positions[val_idx], positions[train_idx]
    knn_local, distances = nearest_neighbors(query, reference, 16)
    knn_global = train_idx[knn_local]
    tri_local, tri_weights, inside = delaunay_neighbors(query, reference)
    tri_global = train_idx[tri_local]
    strategies: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "delaunay": (tri_global, tri_weights),
    }
    for k in (4, 6, 8, 12, 16):
        for power in (1.0, 2.0):
            base = distance_weights(distances[:, :k], power=power)
            strategies[f"knn_k{k}_p{power:g}"] = (knn_global[:, :k], base)
            affine = affine_reproduction_weights(query, reference[knn_local[:, :k]], base)
            strategies[f"affine_k{k}_p{power:g}"] = (knn_global[:, :k], affine)
    # Convex-hull misses use the robust KNN-4/p1 fallback.
    fallback_w = distance_weights(distances[:, :4], power=1.0)
    strategies["delaunay"][0][~inside] = knn_global[~inside, :3]
    strategies["delaunay"][1][~inside] = fallback_w[~inside, :3] / fallback_w[~inside, :3].sum(1, keepdims=True)
    totals = defaultdict(lambda: [0.0, 0, 0.0, 0])
    for qi, target_idx in enumerate(val_idx):
        target_h = np.asarray(channels[target_idx], dtype=np.complex64)
        target_pas = np.abs(np.fft.fft(target_h, axis=0, norm="ortho")) ** 2
        target_pdp = np.abs(np.fft.fft(target_h, axis=-1, norm="ortho")) ** 2
        unique_indices = np.unique(
            np.concatenate([indices[qi] for indices, _ in strategies.values()])
        )
        unique_h = np.asarray(channels[unique_indices], dtype=np.complex64)
        unique_pas = np.abs(np.fft.fft(unique_h, axis=1, norm="ortho")) ** 2
        unique_pdp = np.abs(np.fft.fft(unique_h, axis=-1, norm="ortho")) ** 2
        lookup = {int(global_index): i for i, global_index in enumerate(unique_indices)}
        for name, (indices, all_weights) in strategies.items():
            local = [lookup[int(global_index)] for global_index in indices[qi]]
            source_pas = unique_pas[local]
            source_pdp = unique_pdp[local]
            pred_pas = unit_mix(source_pas, all_weights[qi], axis=1)
            pred_pdp = unit_mix(source_pdp, all_weights[qi], axis=-1)
            ps, pc = cos_sum(pred_pas, target_pas, axis=0)
            ds, dc = cos_sum(pred_pdp, target_pdp, axis=-1)
            totals[name][0] += ps
            totals[name][1] += pc
            totals[name][2] += ds
            totals[name][3] += dc
        if (qi + 1) % 20 == 0:
            print(f"processed {qi + 1}/{len(val_idx)}", flush=True)
    results = {}
    for name, (ps, pc, ds, dc) in totals.items():
        c1, c2 = ps / pc, ds / dc
        results[name] = {"c1_pas": c1, "c2_pdp": c2, "spectral_score": 0.4 * (c1 + c2)}
    for name, metrics in sorted(results.items(), key=lambda item: item[1]["spectral_score"], reverse=True):
        print(name, json.dumps(metrics, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"validation_size": len(val_idx), "inside_hull": int(inside.sum()), "results": results}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
