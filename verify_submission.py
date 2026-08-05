from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from physical_ai.data import RoundData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strictly verify a generated contest submission")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--file", type=Path, default=Path("Round1_Test_Channel.npy"))
    parser.add_argument("--chunk-size", type=int, default=10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    data = RoundData(args.root)
    data.validate()
    channel = np.load(args.file, mmap_mode="r")
    expected_shape = (len(data.test_positions), *data.dims.channel_shape)
    if channel.shape != expected_shape:
        raise ValueError(f"Shape {channel.shape} does not match required {expected_shape}")
    if channel.dtype != np.complex64:
        raise TypeError(f"Expected complex64, got {channel.dtype}")
    finite = True
    zero_elements = 0
    minimum = float("inf")
    maximum = 0.0
    energy = 0.0
    for start in range(0, len(channel), args.chunk_size):
        batch = np.asarray(channel[start : start + args.chunk_size])
        finite &= bool(np.isfinite(batch.real).all() and np.isfinite(batch.imag).all())
        magnitude = np.abs(batch)
        zero_elements += int(np.count_nonzero(magnitude == 0))
        minimum = min(minimum, float(magnitude.min()))
        maximum = max(maximum, float(magnitude.max()))
        energy += float(np.sum(magnitude.astype(np.float64) ** 2))
    if not finite:
        raise ValueError("Submission contains NaN or infinity")
    probe = np.asarray(channel[: min(5, len(channel))])
    pas = np.abs(np.fft.fft(probe, axis=1, norm="ortho")) ** 2
    pdp = np.abs(np.fft.fft(probe, axis=-1, norm="ortho")) ** 2
    pas_min_norm = float(np.linalg.norm(pas, axis=1).min())
    pdp_min_norm = float(np.linalg.norm(pdp, axis=-1).min())
    if pas_min_norm <= 0 or pdp_min_norm <= 0:
        raise ValueError("At least one scored spectrum is identically zero")
    report = {
        "file": str(args.file),
        "file_size": os.path.getsize(args.file),
        "sha256": sha256(args.file),
        "shape": list(channel.shape),
        "dtype": str(channel.dtype),
        "finite": finite,
        "zero_elements": zero_elements,
        "abs_min": minimum,
        "abs_max": maximum,
        "total_energy": energy,
        "probe_pas_min_norm": pas_min_norm,
        "probe_pdp_min_norm": pdp_min_norm,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
