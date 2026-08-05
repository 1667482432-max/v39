from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from physical_ai.data import RoundData
from physical_ai.map_encoder import MapRaster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode the supplied point cloud for Physical AI")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--margin", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/map_raster.npz"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = RoundData(args.root)
    data.validate()
    train = np.asarray(data.train_positions)
    test = np.asarray(data.test_positions)
    all_positions = np.concatenate((train, test, np.asarray(data.dims.bs_position)[None]), axis=0)
    minimum = all_positions[:, :2].min(axis=0) - args.margin
    maximum = all_positions[:, :2].max(axis=0) + args.margin
    raster = MapRaster.from_point_cloud(data.map_path, minimum, maximum, args.resolution)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raster.save(args.output)
    bs = np.asarray(data.dims.bs_position)
    train_context = raster.context_features(train, bs)
    test_context = raster.context_features(test, bs)
    np.savez_compressed(
        args.output.with_name("map_context.npz"), train=train_context, test=test_context
    )
    print(
        f"raster={raster.height.shape}, context={train_context.shape}, "
        f"height_max={raster.height.max():.2f}, occupied={(raster.log_density > 0).mean():.3f}"
    )


if __name__ == "__main__":
    main()
