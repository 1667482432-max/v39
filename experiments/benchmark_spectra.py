from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from physical_ai.data import RoundData
from physical_ai.neighbors import distance_weights, nearest_neighbors


def cosine_sum(prediction: np.ndarray, target: np.ndarray, axis: int) -> tuple[float, int]:
    numerator = np.sum(prediction * target, axis=axis, dtype=np.float64)
    pred_norm = np.sqrt(np.sum(prediction * prediction, axis=axis, dtype=np.float64))
    target_norm = np.sqrt(np.sum(target * target, axis=axis, dtype=np.float64))
    values = numerator / np.maximum(pred_norm * target_norm, 1e-30)
    return float(values.sum(dtype=np.float64)), values.size


def normalized_vectors(power: np.ndarray, axis: int) -> np.ndarray:
    norm = np.sqrt(np.sum(power * power, axis=axis, keepdims=True, dtype=np.float64))
    return power / np.maximum(norm, 1e-30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upper-bound benchmark for KNN PAS/PDP prediction")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--max-k", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("artifacts/spectral_benchmark.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = RoundData(args.root)
    data.validate()
    positions = np.asarray(data.train_positions)
    channels = data.train_channels
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(positions))
    val_idx, train_idx = np.sort(order[: args.validation_size]), np.sort(order[args.validation_size :])
    nn_local, distances = nearest_neighbors(positions[val_idx], positions[train_idx], args.max_k)
    nn_global = train_idx[nn_local]
    candidates = []
    for k in (1, 2, 3, 4, 6, 8, 12):
        if k > args.max_k:
            continue
        for power in ((0.0,) if k == 1 else (0.0, 1.0, 2.0, 4.0, 6.0)):
            for normalized in (False, True):
                candidates.append((k, power, normalized))
    totals: dict[tuple[int, float, bool], list[float]] = defaultdict(lambda: [0.0, 0, 0.0, 0])
    for qi, target_global in enumerate(val_idx):
        neighbor_h = np.asarray(channels[nn_global[qi]], dtype=np.complex64)
        target_h = np.asarray(channels[target_global], dtype=np.complex64)
        neighbor_pas = np.abs(np.fft.fft(neighbor_h, axis=1, norm="ortho")) ** 2
        target_pas = np.abs(np.fft.fft(target_h, axis=0, norm="ortho")) ** 2
        neighbor_pdp = np.abs(np.fft.fft(neighbor_h, axis=-1, norm="ortho")) ** 2
        target_pdp = np.abs(np.fft.fft(target_h, axis=-1, norm="ortho")) ** 2
        for key in candidates:
            k, power, normalized = key
            if k == 1:
                w = np.ones(1)
            elif power == 0:
                w = np.full(k, 1.0 / k)
            else:
                w = distance_weights(distances[qi : qi + 1, :k], power=power)[0]
            pas_source = neighbor_pas[:k]
            pdp_source = neighbor_pdp[:k]
            if normalized:
                pas_source = normalized_vectors(pas_source, axis=1)
                pdp_source = normalized_vectors(pdp_source, axis=-1)
            pred_pas = np.einsum("k,kmns->mns", w, pas_source, optimize=True)
            pred_pdp = np.einsum("k,kmns->mns", w, pdp_source, optimize=True)
            ps, pc = cosine_sum(pred_pas, target_pas, axis=0)
            ds, dc = cosine_sum(pred_pdp, target_pdp, axis=-1)
            totals[key][0] += ps
            totals[key][1] += pc
            totals[key][2] += ds
            totals[key][3] += dc
        if (qi + 1) % 20 == 0:
            print(f"processed {qi + 1}/{len(val_idx)}", flush=True)
    results = {}
    for (k, power, normalized), (ps, pc, ds, dc) in totals.items():
        c1, c2 = ps / pc, ds / dc
        name = f"k{k}_p{power:g}_{'unit' if normalized else 'raw'}"
        results[name] = {"c1_pas": c1, "c2_pdp": c2, "spectral_score": 0.4 * (c1 + c2)}
    for name, values in sorted(results.items(), key=lambda item: item[1]["spectral_score"], reverse=True)[:15]:
        print(name, json.dumps(values, sort_keys=True))
    payload = {"validation_size": len(val_idx), "seed": args.seed, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
