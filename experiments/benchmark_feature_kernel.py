from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from physical_ai.data import RoundData
from physical_ai.features import SpectralFeatureLayout, nonzero_feature_indices


def pairwise_distance(
    queries: np.ndarray,
    references: np.ndarray,
    mode: str,
    ratio: float,
    bs_xy: np.ndarray,
) -> np.ndarray:
    if mode == "cartesian":
        delta = queries[:, None, :2] - references[None, :, :2]
        delta[..., 0] *= ratio
        return np.sqrt(np.sum(delta * delta, axis=-1))
    qrel, rrel = queries[:, :2] - bs_xy, references[:, :2] - bs_xy
    qr, rr = np.linalg.norm(qrel, axis=1), np.linalg.norm(rrel, axis=1)
    qt, rt = np.arctan2(qrel[:, 1], qrel[:, 0]), np.arctan2(rrel[:, 1], rrel[:, 0])
    dt = (qt[:, None] - rt[None, :] + np.pi) % (2 * np.pi) - np.pi
    arc = dt * (qr[:, None] + rr[None, :]) * 0.5
    radial = qr[:, None] - rr[None, :]
    return np.sqrt((ratio * radial) ** 2 + arc**2)


def cosine_means(prediction: np.ndarray, target: np.ndarray, layout: SpectralFeatureLayout) -> tuple[float, float]:
    pas_p = prediction[:, : layout.pas_size].reshape(len(prediction), 256, 4)
    pas_t = target[:, : layout.pas_size].reshape(len(target), 256, 4)
    pdp_p = prediction[:, layout.pas_size :].reshape(len(prediction), 2, 4, 192)
    pdp_t = target[:, layout.pas_size :].reshape(len(target), 2, 4, 192)
    pas = np.sum(pas_p * pas_t, axis=1) / np.maximum(
        np.linalg.norm(pas_p, axis=1) * np.linalg.norm(pas_t, axis=1), 1e-30
    )
    pdp = np.sum(pdp_p * pdp_t, axis=-1) / np.maximum(
        np.linalg.norm(pdp_p, axis=-1) * np.linalg.norm(pdp_t, axis=-1), 1e-30
    )
    return float(pas.mean()), float(pdp.mean())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune score-aligned spatial kernels")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--features", type=Path, default=Path("artifacts/spectral_features.npy"))
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output", type=Path, default=Path("artifacts/kernel_benchmark.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = RoundData(args.root)
    data.validate()
    all_features = np.load(args.features, mmap_mode="r")
    valid = nonzero_feature_indices(all_features)
    features = np.asarray(all_features[valid])
    positions = np.asarray(data.train_positions)[valid]
    layout = SpectralFeatureLayout.from_dimensions(data.dims)
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(positions))
    val_idx, train_idx = np.sort(permutation[: args.validation_size]), np.sort(permutation[args.validation_size :])
    target = np.asarray(features[val_idx])
    results = {}
    for mode in ("cartesian", "polar"):
        for ratio in (0.5, 0.75, 1.0, 1.5, 2.0):
            distances = pairwise_distance(
                positions[val_idx], positions[train_idx], mode, ratio, np.asarray(data.dims.bs_position[:2])
            )
            order = np.argsort(distances, axis=1)
            for k in (4, 8, 16, 24, 32, 48, 64):
                idx = order[:, :k]
                d = np.take_along_axis(distances, idx, axis=1)
                source = np.asarray(features[train_idx[idx]])
                for power in (1.0, 2.0, 3.0, 4.0):
                    weights = np.maximum(d, 1e-6) ** (-power)
                    weights /= weights.sum(axis=1, keepdims=True)
                    prediction = np.einsum("qk,qkd->qd", weights, source, optimize=True)
                    c1, c2 = cosine_means(prediction, target, layout)
                    name = f"{mode}_r{ratio:g}_k{k}_p{power:g}"
                    results[name] = {"feature_pas": c1, "feature_pdp": c2, "mean": 0.5 * (c1 + c2)}
    for bandwidth in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0):
        distances = pairwise_distance(
            positions[val_idx], positions[train_idx], "cartesian", 1.0, np.asarray(data.dims.bs_position[:2])
        )
        order = np.argsort(distances, axis=1)[:, :64]
        d = np.take_along_axis(distances, order, axis=1)
        source = np.asarray(features[train_idx[order]])
        weights = np.exp(-0.5 * (d / bandwidth) ** 2)
        weights /= weights.sum(axis=1, keepdims=True)
        prediction = np.einsum("qk,qkd->qd", weights, source, optimize=True)
        c1, c2 = cosine_means(prediction, target, layout)
        results[f"gaussian_h{bandwidth:g}"] = {
            "feature_pas": c1,
            "feature_pdp": c2,
            "mean": 0.5 * (c1 + c2),
        }
    print("TOP PAS")
    for name, values in sorted(results.items(), key=lambda x: x[1]["feature_pas"], reverse=True)[:10]:
        print(name, json.dumps(values))
    print("TOP PDP")
    for name, values in sorted(results.items(), key=lambda x: x[1]["feature_pdp"], reverse=True)[:10]:
        print(name, json.dumps(values))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"validation_size": len(val_idx), "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
