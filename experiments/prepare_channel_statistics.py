from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from physical_ai.data import RoundData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract compact physical channel statistics")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/channel_group_energy.npy")
    )
    parser.add_argument(
        "--detailed-output",
        type=Path,
        default=Path("artifacts/channel_power_marginals.npz"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = RoundData(args.root)
    data.validate()
    channels = data.train_channels
    result = np.empty(
        (len(data.train_positions), data.dims.bs_polarizations, data.dims.ue_antennas),
        dtype=np.float64,
    )
    antenna_ue = np.empty(
        (len(data.train_positions), data.dims.bs_antennas, data.dims.ue_antennas),
        dtype=np.float64,
    )
    ue_subcarrier = np.empty(
        (len(data.train_positions), data.dims.ue_antennas, data.dims.subcarriers),
        dtype=np.float64,
    )
    for start in range(0, len(result), args.batch_size):
        stop = min(start + args.batch_size, len(result))
        channel = np.asarray(channels[start:stop])
        grouped = channel.reshape(
            stop - start,
            data.dims.bs_polarizations,
            data.dims.bs_h * data.dims.bs_v,
            data.dims.ue_antennas,
            data.dims.subcarriers,
        )
        result[start:stop] = np.sum(np.abs(grouped) ** 2, axis=(2, 4), dtype=np.float64)
        power = np.abs(channel) ** 2
        antenna_ue[start:stop] = np.sum(power, axis=3, dtype=np.float64)
        ue_subcarrier[start:stop] = np.sum(power, axis=1, dtype=np.float64)
        print(f"processed {stop}/{len(result)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result)
    np.savez_compressed(
        args.detailed_output,
        group=result,
        antenna_ue=antenna_ue,
        ue_subcarrier=ue_subcarrier,
    )
    print(f"saved {args.output} shape={result.shape}")


if __name__ == "__main__":
    main()
