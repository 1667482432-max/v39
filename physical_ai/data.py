from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RoundDimensions:
    train_points: int
    test_points: int
    bs_antennas: int
    bs_h: int
    bs_v: int
    bs_polarizations: int
    ue_antennas: int
    ue_h: int
    ue_v: int
    ue_polarizations: int
    subcarriers: int
    iq_components: int
    bs_position: tuple[float, float, float]
    score_weights: tuple[float, float, float]

    @classmethod
    def from_json(cls, path: str | Path) -> "RoundDimensions":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls(
            train_points=int(raw["P_Train"]),
            test_points=int(raw["P_Test"]),
            bs_antennas=int(raw["M"]),
            bs_h=int(raw["M_H"]),
            bs_v=int(raw["M_V"]),
            bs_polarizations=int(raw["M_P"]),
            ue_antennas=int(raw["N"]),
            ue_h=int(raw["N_H"]),
            ue_v=int(raw["N_V"]),
            ue_polarizations=int(raw["N_P"]),
            subcarriers=int(raw["S"]),
            iq_components=int(raw["Q"]),
            bs_position=tuple(float(x) for x in raw["X"]),
            score_weights=tuple(float(x) for x in raw["w"]),
        )

    @property
    def channel_shape(self) -> tuple[int, int, int]:
        return (self.bs_antennas, self.ue_antennas, self.subcarriers)

    def validate_antenna_layout(self) -> None:
        if self.bs_antennas != self.bs_polarizations * self.bs_h * self.bs_v:
            raise ValueError("M must equal M_P * M_H * M_V")
        if self.ue_antennas != self.ue_polarizations * self.ue_h * self.ue_v:
            raise ValueError("N must equal N_P * N_H * N_V")


@dataclass
class RoundData:
    root: Path
    round_name: str = "Round1"

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.dims = RoundDimensions.from_json(self.root / f"{self.round_name}_Setup.json")
        self.dims.validate_antenna_layout()

    @property
    def train_positions(self) -> np.ndarray:
        return np.load(self.root / f"{self.round_name}_Train_Pos.npy", mmap_mode="r")

    @property
    def test_positions(self) -> np.ndarray:
        return np.load(self.root / f"{self.round_name}_Test_Pos.npy", mmap_mode="r")

    @property
    def train_channels(self) -> np.ndarray:
        return np.load(self.root / f"{self.round_name}_Train_Channel.npy", mmap_mode="r")

    @property
    def map_path(self) -> Path:
        return self.root / f"{self.round_name}_Map.ply"

    @property
    def output_path(self) -> Path:
        return self.root / f"{self.round_name}_Test_Channel.npy"

    def validate(self) -> None:
        train_pos = self.train_positions
        test_pos = self.test_positions
        channels = self.train_channels
        if train_pos.ndim != 2 or train_pos.shape[1] != 3:
            raise ValueError(f"Bad training-position shape: {train_pos.shape}")
        if test_pos.ndim != 2 or test_pos.shape[1] != 3:
            raise ValueError(f"Bad test-position shape: {test_pos.shape}")
        expected_tail = self.dims.channel_shape
        if channels.shape[1:] != expected_tail:
            raise ValueError(f"Bad channel shape {channels.shape}; expected (*, {expected_tail})")
        if channels.shape[0] != train_pos.shape[0]:
            raise ValueError("Training positions and channels have different sample counts")
        if not np.issubdtype(channels.dtype, np.complexfloating):
            raise TypeError(f"Channels must be complex, got {channels.dtype}")
        if train_pos.shape[0] != self.dims.train_points:
            warnings.warn(
                f"Setup declares P_Train={self.dims.train_points}, but arrays contain "
                f"{train_pos.shape[0]}; using the authoritative array length.",
                stacklevel=2,
            )
        if test_pos.shape[0] != self.dims.test_points:
            warnings.warn(
                f"Setup declares P_Test={self.dims.test_points}, but array contains "
                f"{test_pos.shape[0]}; using the authoritative array length.",
                stacklevel=2,
            )


def reshape_bs_layout(channel: np.ndarray, dims: RoundDimensions) -> np.ndarray:
    """Expose the mandated flattened order: polarization -> H -> V."""
    if channel.shape[-3] != dims.bs_antennas:
        raise ValueError("Unexpected flattened BS antenna dimension")
    lead = channel.shape[:-3]
    return channel.reshape(
        *lead,
        dims.bs_polarizations,
        dims.bs_h,
        dims.bs_v,
        dims.ue_antennas,
        dims.subcarriers,
    )
