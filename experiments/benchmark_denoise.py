from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter, uniform_filter1d

from physical_ai.data import RoundData
from physical_ai.neighbors import distance_weights, nearest_neighbors


def normalized_mix(source: np.ndarray, weights: np.ndarray, axis: int) -> np.ndarray:
    norm = np.sqrt(np.sum(source * source, axis=axis, keepdims=True, dtype=np.float64))
    unit = source / np.maximum(norm, 1e-30)
    return np.einsum("k,kmns->mns", weights, unit, optimize=True)


def cosine_sum(pred: np.ndarray, target: np.ndarray, axis: int) -> tuple[float, int]:
    numerator = np.sum(pred * target, axis=axis, dtype=np.float64)
    denominator = np.sqrt(np.sum(pred * pred, axis=axis, dtype=np.float64)) * np.sqrt(
        np.sum(target * target, axis=axis, dtype=np.float64)
    )
    value = numerator / np.maximum(denominator, 1e-30)
    return float(value.sum()), value.size


def smooth_pdp_array(pdp: np.ndarray, h_window: int, v_window: int) -> np.ndarray:
    shaped = pdp.reshape(2, 16, 8, 4, 192)
    if h_window == 16 and v_window == 8:
        return np.broadcast_to(shaped.mean(axis=(1, 2), keepdims=True), shaped.shape).reshape(pdp.shape)
    smoothed = uniform_filter(
        shaped, size=(1, h_window, v_window, 1, 1), mode="nearest"
    )
    return smoothed.reshape(pdp.shape)


def grouped_average(power: np.ndarray, kind: str) -> np.ndarray:
    if kind.startswith("pas"):
        shaped = power.reshape(256, 2, 1, 2, 192)
        if "s" in kind:
            shaped = np.broadcast_to(shaped.mean(axis=-1, keepdims=True), shaped.shape)
        if "uev" in kind:
            shaped = np.broadcast_to(shaped.mean(axis=3, keepdims=True), shaped.shape)
        if "ueall" in kind:
            mean = shaped.mean(axis=(1, 2, 3), keepdims=True)
            shaped = np.broadcast_to(mean, shaped.shape)
        return shaped.reshape(power.shape)
    shaped = power.reshape(2, 16, 8, 2, 1, 2, 192)
    if "bsarray" in kind:
        shaped = np.broadcast_to(shaped.mean(axis=(1, 2), keepdims=True), shaped.shape)
    if "bsall" in kind:
        shaped = np.broadcast_to(shaped.mean(axis=(0, 1, 2), keepdims=True), shaped.shape)
    if "uev" in kind:
        shaped = np.broadcast_to(shaped.mean(axis=5, keepdims=True), shaped.shape)
    if "ueall" in kind:
        shaped = np.broadcast_to(shaped.mean(axis=(3, 4, 5), keepdims=True), shaped.shape)
    return shaped.reshape(power.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Physical-axis denoising benchmark")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output", type=Path, default=Path("artifacts/denoise_benchmark.json"))
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
    nn_local, distances = nearest_neighbors(positions[val_idx], positions[train_idx], 16)
    nn_global = train_idx[nn_local]
    pas_weights = distance_weights(distances[:, :16], power=2.0)
    pdp_weights = distance_weights(distances[:, :4], power=1.0)
    pas_windows = (1, 3, 5, 9, 17, 33, 65, 129, 192)
    pdp_windows = ((1, 1), (3, 3), (5, 3), (7, 5), (16, 8))
    pas_groups = ("pas_s", "pas_s_uev", "pas_s_ueall", "pas_uev")
    pdp_groups = (
        "pdp_bsarray",
        "pdp_bsall",
        "pdp_bsarray_uev",
        "pdp_bsarray_ueall",
        "pdp_bsall_uev",
        "pdp_bsall_ueall",
    )
    totals = defaultdict(lambda: [0.0, 0])
    for qi, target_idx in enumerate(val_idx):
        source_h = np.asarray(channels[nn_global[qi]], dtype=np.complex64)
        target_h = np.asarray(channels[target_idx], dtype=np.complex64)
        source_pas = np.abs(np.fft.fft(source_h, axis=1, norm="ortho")) ** 2
        source_pdp = np.abs(np.fft.fft(source_h[:4], axis=-1, norm="ortho")) ** 2
        target_pas = np.abs(np.fft.fft(target_h, axis=0, norm="ortho")) ** 2
        target_pdp = np.abs(np.fft.fft(target_h, axis=-1, norm="ortho")) ** 2
        pred_pas = normalized_mix(source_pas, pas_weights[qi], axis=1)
        pred_pdp = normalized_mix(source_pdp, pdp_weights[qi], axis=-1)
        for window in pas_windows:
            if window == 1:
                smoothed = pred_pas
            elif window == 192:
                smoothed = np.broadcast_to(pred_pas.mean(axis=-1, keepdims=True), pred_pas.shape)
            else:
                smoothed = uniform_filter1d(pred_pas, size=window, axis=-1, mode="nearest")
            value, count = cosine_sum(smoothed, target_pas, axis=0)
            totals[f"pas_s{window}"][0] += value
            totals[f"pas_s{window}"][1] += count
        for h_window, v_window in pdp_windows:
            smoothed = smooth_pdp_array(pred_pdp, h_window, v_window)
            value, count = cosine_sum(smoothed, target_pdp, axis=-1)
            totals[f"pdp_h{h_window}_v{v_window}"][0] += value
            totals[f"pdp_h{h_window}_v{v_window}"][1] += count
        for group in pas_groups:
            smoothed = grouped_average(pred_pas, group)
            value, count = cosine_sum(smoothed, target_pas, axis=0)
            totals[group][0] += value
            totals[group][1] += count
        for group in pdp_groups:
            smoothed = grouped_average(pred_pdp, group)
            value, count = cosine_sum(smoothed, target_pdp, axis=-1)
            totals[group][0] += value
            totals[group][1] += count
        if (qi + 1) % 20 == 0:
            print(f"processed {qi + 1}/{len(val_idx)}", flush=True)
    results = {name: value / count for name, (value, count) in totals.items()}
    for name, value in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print(name, f"{value:.9f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"validation_size": len(val_idx), "results": results}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
