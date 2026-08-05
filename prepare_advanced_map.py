from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from physical_ai.advanced_map import AdvancedMapRaster
from physical_ai.data import RoundData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normal-aware Physical-AI map features")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--margin", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/map_raster_advanced.npz"))
    parser.add_argument(
        "--rebuild-raster",
        action="store_true",
        help="Rebuild the point-cloud raster even when --output already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = RoundData(args.root)
    data.validate()
    train = np.asarray(data.train_positions)
    test = np.asarray(data.test_positions)
    bs = np.asarray(data.dims.bs_position)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.rebuild_raster:
        raster = AdvancedMapRaster.load(args.output)
        print(f"reusing raster {args.output}", flush=True)
    else:
        all_positions = np.concatenate((train, test, bs[None]), axis=0)
        minimum = all_positions[:, :2].min(0) - args.margin
        maximum = all_positions[:, :2].max(0) + args.margin
        raster = AdvancedMapRaster.from_point_cloud(
            data.map_path, minimum, maximum, args.resolution
        )
        raster.save(args.output)
    # One pass avoids recomputing all multiscale Gaussian fields for train/test.
    sample_positions = np.concatenate((train, test), axis=0)
    advanced = raster.context_features(sample_positions, bs)
    train_advanced, test_advanced = np.split(advanced, [len(train)])
    base = np.load(args.output.with_name("map_context.npz"))
    train_context = np.concatenate((base["train"], train_advanced), axis=1)
    test_context = np.concatenate((base["test"], test_advanced), axis=1)
    context_path = args.output.with_name("map_context_advanced.npz")
    np.savez_compressed(context_path, train=train_context, test=test_context)
    print(
        f"raster={raster.height.shape}, advanced={train_advanced.shape}, "
        f"context={train_context.shape}, wall_mean={raster.wall_log_density.mean():.3f}"
    )


if __name__ == "__main__":
    main()
