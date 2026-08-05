from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from experiments.search_advanced_map_features import BLOCKS, standardized
from physical_ai.features import nonzero_feature_indices


EXTRA_BLOCKS = {
    "material_center_multiscale": (slice(54, 58), slice(201, 209)),
    "center_physics_multiscale": (
        slice(18, 24),
        slice(54, 58),
        slice(201, 209),
    ),
    "corridor_material_multiscale": (
        slice(0, 70),
        slice(201, 209),
    ),
    "endpoint_multiscale": (slice(70, 105), slice(201, 209)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search advanced physical embeddings for channel-group energy interpolation"
    )
    parser.add_argument(
        "--context", type=Path, default=Path("artifacts/map_context_advanced.npz")
    )
    parser.add_argument(
        "--energy", type=Path, default=Path("artifacts/channel_group_energy.npy")
    )
    parser.add_argument("--neighbors", type=int, nargs="+", default=[32, 48, 64, 96])
    parser.add_argument("--powers", type=float, nargs="+", default=[2.0, 3.0, 4.0, 5.0])
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/advanced_group_energy_search.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contexts_all = np.load(args.context)["train"].astype(np.float32)
    positions_all = np.load("Round1_Train_Pos.npy").astype(np.float32)
    features = np.load("artifacts/spectral_features.npy", mmap_mode="r")
    valid_global = nonzero_feature_indices(features)
    inverse = np.full(len(features), -1, dtype=np.int64)
    inverse[valid_global] = np.arange(len(valid_global))
    contexts = contexts_all[valid_global]
    xy = positions_all[valid_global, :2].astype(np.float64)
    advanced = contexts[:, 153:]
    legacy = np.concatenate(
        (standardized(contexts[:, 103:128]), standardized(contexts[:, 128:153])), axis=1
    )
    blocks = {name: (block,) for name, block in BLOCKS.items()}
    blocks.update(EXTRA_BLOCKS)
    embeddings = {}
    for name, slices in blocks.items():
        raw = np.concatenate([advanced[:, block] for block in slices], axis=1)
        block = standardized(raw)
        for scale in args.scales:
            embeddings[f"base+{name}_s{scale:g}"] = np.concatenate(
                (xy, 3.0 * legacy, scale * block), axis=1
            )

    energy = np.load(args.energy)
    fraction = energy / np.maximum(energy.sum(axis=(1, 2), keepdims=True), 1e-30)
    checkpoints = [
        Path(f"artifacts/cv_noout_split{suffix}.pt")
        for suffix in ("101", "202", "20260804", "303", "404")
    ]
    totals: dict[str, list[float]] = {}
    maximum_neighbors = max(args.neighbors)
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        train_idx = inverse[np.asarray(checkpoint["train_indices"], dtype=np.int64)]
        val_idx = inverse[np.asarray(checkpoint["validation_indices"], dtype=np.int64)]
        target = fraction[valid_global[val_idx]].clip(1e-12)
        target /= target.sum(axis=(1, 2), keepdims=True)
        for embedding_name, embedding in embeddings.items():
            distance, local = cKDTree(embedding[train_idx]).query(
                embedding[val_idx], k=maximum_neighbors, workers=-1
            )
            source = fraction[valid_global[train_idx[local]]].clip(1e-12)
            for neighbors in args.neighbors:
                local_distance = distance[:, :neighbors]
                local_source = source[:, :neighbors]
                for power in args.powers:
                    weight = (local_distance + 1e-3) ** (-power)
                    weight /= weight.sum(axis=1, keepdims=True)
                    prediction = np.exp(
                        np.sum(weight[:, :, None, None] * np.log(local_source), axis=1)
                    )
                    prediction /= prediction.sum(axis=(1, 2), keepdims=True)
                    log_error = np.log(prediction.clip(1e-12)) - np.log(target)
                    log_rmse = float(np.sqrt(np.mean(log_error * log_error)))
                    kl = float(np.mean(np.sum(target * -log_error, axis=(1, 2))))
                    cosine = float(
                        np.mean(
                            np.sum(prediction * target, axis=(1, 2))
                            / np.maximum(
                                np.linalg.norm(prediction.reshape(len(prediction), -1), axis=1)
                                * np.linalg.norm(target.reshape(len(target), -1), axis=1),
                                1e-12,
                            )
                        )
                    )
                    key = f"{embedding_name}__k{neighbors}_p{power:g}"
                    item = totals.setdefault(key, [0.0, 0.0, 0.0, 0.0])
                    item[0] += log_rmse
                    item[1] += kl
                    item[2] += cosine
                    item[3] += 1.0
        print(checkpoint_path.stem, flush=True)

    results = {
        key: {
            "log_rmse": value[0] / value[3],
            "kl": value[1] / value[3],
            "cosine": value[2] / value[3],
        }
        for key, value in totals.items()
    }
    top = {
        "log_rmse": sorted(results.items(), key=lambda item: item[1]["log_rmse"])[:30],
        "kl": sorted(results.items(), key=lambda item: item[1]["kl"])[:30],
        "cosine": sorted(
            results.items(), key=lambda item: item[1]["cosine"], reverse=True
        )[:30],
    }
    output = {"top": top, "results": results}
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({name: rows[:10] for name, rows in top.items()}, indent=2))


if __name__ == "__main__":
    main()
