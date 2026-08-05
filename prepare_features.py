from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from physical_ai.data import RoundData
from physical_ai.features import SpectralFeatureLayout, compact_spectral_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract compact score-aligned channel features")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifacts/spectral_features.npy"))
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = RoundData(args.root)
    data.validate()
    channels = data.train_channels
    layout = SpectralFeatureLayout.from_dimensions(data.dims)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        args.output, mode="w+", dtype=np.float32, shape=(len(channels), layout.total_size)
    )
    for start in range(0, len(channels), args.batch_size):
        stop = min(start + args.batch_size, len(channels))
        batch = torch.from_numpy(np.array(channels[start:stop], dtype=np.complex64, copy=True))
        output[start:stop] = compact_spectral_features(batch, data.dims).numpy()
        if stop % 100 == 0 or stop == len(channels):
            output.flush()
            print(f"processed {stop}/{len(channels)}", flush=True)
    metadata = {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "pas_size": layout.pas_size,
        "pdp_size": layout.pdp_size,
        "description": "normalized mean PAS(M,N) followed by polarization-mean PDP(P,N,S)",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
