from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.search_kriging_ensemble_gpu import Config, predict_config
from experiments.search_spatial_kernels_gpu import cosine_parts
from physical_ai.features import SpectralFeatureLayout, nonzero_feature_indices


FOCUSED_KERNEL = Config("advanced", "exponential", 32, 0.5, 0.05, False)
EXTRA_KERNELS = (
    Config("advanced", "exponential", 16, 0.5, 0.01, True),
    Config("advanced", "exponential", 24, 0.75, 0.05, False),
)

# Offsets are relative to the 153-dimensional legacy context.  Small physical
# sub-blocks make it possible to reject noisy parts without rerunning old
# geometry/map searches.
BLOCKS = {
    "corridor_center": slice(18, 24),
    "corridor_core": slice(12, 30),
    "corridor": slice(0, 42),
    "material_center": slice(54, 58),
    "material_core": slice(50, 62),
    "material": slice(42, 70),
    "corridor_material": slice(0, 70),
    "endpoint_near": slice(70, 91),
    "endpoint_far": slice(91, 105),
    "endpoint": slice(70, 105),
    "skyline_height": slice(105, 153),
    "skyline_wall": slice(153, 201),
    "skyline": slice(105, 201),
    "endpoint_skyline": slice(70, 201),
    "multiscale": slice(201, 209),
    "advanced": slice(0, 209),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incremental search over newly extracted point-cloud physics features"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260804, 101, 202, 303, 404])
    parser.add_argument("--validation-size", type=int, default=400)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 3.0, 4.0])
    parser.add_argument(
        "--blocks",
        nargs="+",
        default=None,
        help="Optional subset of advanced block names for a focused follow-up",
    )
    parser.add_argument("--all-kernels", action="store_true")
    parser.add_argument("--context", type=Path, default=Path("artifacts/map_context_advanced.npz"))
    parser.add_argument(
        "--baseline-results", type=Path, default=Path("artifacts/map_metric_kriging_5fold.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/advanced_map_features_5fold.json")
    )
    return parser.parse_args()


def standardized(raw: np.ndarray) -> np.ndarray:
    value = (raw - raw.mean(0, keepdims=True)) / np.maximum(raw.std(0, keepdims=True), 1e-3)
    return value / np.sqrt(value.shape[1])


def baseline_score(
    path: Path, seeds: list[int]
) -> tuple[str, dict[str, float]] | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    # The stored top score is a fixed-candidate average over every historical
    # fold.  Do not compare it with a pilot that uses only a subset of folds.
    if set(map(str, seeds)) != set(raw.get("fold_bests", {})):
        return None
    name, score = raw["top"]["mean"][0]
    return name, score


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    positions_all = np.load("Round1_Train_Pos.npy").astype(np.float32)
    contexts_all = np.load(args.context)["train"].astype(np.float32)
    features_all = np.asarray(
        np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32
    )
    if contexts_all.shape[1] != 153 + 209:
        raise ValueError(f"Expected 362 context columns, got {contexts_all.shape[1]}")
    valid = nonzero_feature_indices(features_all)
    positions = positions_all[valid]
    contexts = contexts_all[valid]
    features_np = features_all[valid]
    features = torch.from_numpy(features_np).to("cuda")
    xy = positions[:, :2].astype(np.float64)
    base = np.concatenate(
        (
            xy,
            3.0 * standardized(contexts[:, 103:128]),
            3.0 * standardized(contexts[:, 128:153]),
        ),
        axis=1,
    )
    advanced = contexts[:, 153:]
    raw_blocks = {
        name: advanced[:, block] for name, block in BLOCKS.items()
    }
    raw_blocks.update(
        {
            "material_center_endpoint_near": np.concatenate(
                (advanced[:, 54:58], advanced[:, 70:91]), axis=1
            ),
            "material_center_multiscale": np.concatenate(
                (advanced[:, 54:58], advanced[:, 201:209]), axis=1
            ),
            "material_center_corridor": np.concatenate(
                (advanced[:, 54:58], advanced[:, 12:30]), axis=1
            ),
        }
    )
    selected_blocks = args.blocks or list(BLOCKS)
    unknown = sorted(set(selected_blocks) - set(raw_blocks))
    if unknown:
        raise ValueError(f"Unknown advanced blocks: {unknown}")
    standardized_blocks = {
        name: standardized(advanced[:, block]) for name, block in BLOCKS.items()
        if name in selected_blocks
    }
    for name in selected_blocks:
        if name not in standardized_blocks:
            standardized_blocks[name] = standardized(raw_blocks[name])
    embeddings = {
        f"base+{name}_s{scale:g}": np.concatenate((base, scale * value), axis=1)
        for name, value in standardized_blocks.items()
        for scale in args.scales
    }
    kernels = (FOCUSED_KERNEL,) + (EXTRA_KERNELS if args.all_kernels else ())
    layout = SpectralFeatureLayout(1024, 1536)
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    fold_bests = {}
    for seed in args.seeds:
        permutation = np.random.default_rng(seed).permutation(len(features_np))
        query_idx = np.sort(permutation[: args.validation_size])
        train_idx = np.sort(permutation[args.validation_size :])
        target = features[query_idx]
        fold_results = {}
        for embedding_name, embedding in embeddings.items():
            for kernel in kernels:
                prediction = predict_config(kernel, embedding, train_idx, query_idx, features)
                pas, pdp = cosine_parts(prediction, target, layout)
                name = f"{embedding_name}__{kernel.name}"
                score = {"pas": pas, "pdp": pdp, "mean": 0.5 * (pas + pdp)}
                fold_results[name] = score
                totals[name][0] += score["pas"]
                totals[name][1] += score["pdp"]
                totals[name][2] += score["mean"]
        fold_bests[str(seed)] = {
            metric: max(fold_results.items(), key=lambda item: item[1][metric])
            for metric in ("pas", "pdp", "mean")
        }
        print(seed, json.dumps(fold_bests[str(seed)]), flush=True)
    count = len(args.seeds)
    summary = {
        name: {"pas": values[0] / count, "pdp": values[1] / count, "mean": values[2] / count}
        for name, values in totals.items()
    }
    top = {
        metric: sorted(summary.items(), key=lambda item: item[1][metric], reverse=True)[:50]
        for metric in ("pas", "pdp", "mean")
    }
    baseline = baseline_score(args.baseline_results, args.seeds)
    comparison = None
    if baseline is not None:
        best_name, best = top["mean"][0]
        comparison = {
            "historical_name": baseline[0],
            "historical": baseline[1],
            "advanced_name": best_name,
            "advanced": best,
            "delta": {key: best[key] - baseline[1][key] for key in ("pas", "pdp", "mean")},
        }
    result = {
        "context": str(args.context),
        "kernels": [kernel.name for kernel in kernels],
        "fold_bests": fold_bests,
        "summary": summary,
        "top": top,
        "comparison": comparison,
    }
    print("TOP", json.dumps({key: value[:10] for key, value in top.items()}, indent=2))
    print("COMPARISON", json.dumps(comparison, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
