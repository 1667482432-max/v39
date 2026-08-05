from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import RoundDimensions


@dataclass(frozen=True)
class SpectralFeatureLayout:
    pas_size: int
    pdp_size: int

    @classmethod
    def from_dimensions(cls, dims: RoundDimensions) -> "SpectralFeatureLayout":
        return cls(
            pas_size=dims.bs_antennas * dims.ue_antennas,
            pdp_size=dims.bs_polarizations * dims.ue_antennas * dims.subcarriers,
        )

    @property
    def total_size(self) -> int:
        return self.pas_size + self.pdp_size


@torch.inference_mode()
def compact_spectral_features(
    channel: torch.Tensor,
    dims: RoundDimensions,
    epsilon: float = 1e-20,
) -> torch.Tensor:
    """Compress one or more channels into score-aligned physical spectra.

    Input is ``(..., M, N, S)``. PAS is normalized per ``(N,S)`` vector and
    averaged over subcarriers. PDP is normalized per ``(M,N)`` delay vector and
    averaged across the H/V elements within each BS polarization.
    """
    if channel.shape[-3:] != dims.channel_shape:
        raise ValueError(f"Expected channel tail {dims.channel_shape}, got {channel.shape[-3:]}")
    lead = channel.shape[:-3]
    angular_power = torch.abs(torch.fft.fft(channel, dim=-3, norm="ortho")).square()
    angular_power = angular_power / torch.linalg.vector_norm(
        angular_power, dim=-3, keepdim=True
    ).clamp_min(epsilon)
    pas = angular_power.mean(dim=-1)
    pas = pas / torch.linalg.vector_norm(pas, dim=-2, keepdim=True).clamp_min(epsilon)

    delay_power = torch.abs(torch.fft.fft(channel, dim=-1, norm="ortho")).square()
    delay_power = delay_power / torch.linalg.vector_norm(
        delay_power, dim=-1, keepdim=True
    ).clamp_min(epsilon)
    pdp = delay_power.reshape(
        *lead,
        dims.bs_polarizations,
        dims.bs_h,
        dims.bs_v,
        dims.ue_antennas,
        dims.subcarriers,
    ).mean(dim=(-4, -3))
    pdp = pdp / torch.linalg.vector_norm(pdp, dim=-1, keepdim=True).clamp_min(epsilon)
    return torch.cat((pas.reshape(*lead, -1), pdp.reshape(*lead, -1)), dim=-1)


def split_spectral_features(
    features: torch.Tensor, dims: RoundDimensions
) -> tuple[torch.Tensor, torch.Tensor]:
    layout = SpectralFeatureLayout.from_dimensions(dims)
    if features.shape[-1] != layout.total_size:
        raise ValueError(f"Expected {layout.total_size} features, got {features.shape[-1]}")
    pas = features[..., : layout.pas_size].reshape(
        *features.shape[:-1], dims.bs_antennas, dims.ue_antennas
    )
    pdp = features[..., layout.pas_size :].reshape(
        *features.shape[:-1], dims.bs_polarizations, dims.ue_antennas, dims.subcarriers
    )
    return pas, pdp


def spectral_targets_from_features(
    features: torch.Tensor,
    dims: RoundDimensions,
    epsilon: float = 1e-20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand compact Physical-AI features into compatible PAS/PDP powers."""
    pas_feature, pdp_feature = split_spectral_features(features, dims)
    pas = pas_feature[..., None].expand(
        *pas_feature.shape, dims.subcarriers
    )
    pas = pas / pas.sum(dim=-3, keepdim=True).clamp_min(epsilon)
    pdp = pdp_feature.reshape(
        *pdp_feature.shape[:-3],
        dims.bs_polarizations,
        1,
        1,
        dims.ue_antennas,
        dims.subcarriers,
    ).expand(
        *pdp_feature.shape[:-3],
        dims.bs_polarizations,
        dims.bs_h,
        dims.bs_v,
        dims.ue_antennas,
        dims.subcarriers,
    ).reshape(*pdp_feature.shape[:-3], *dims.channel_shape)
    pdp = pdp / pdp.sum(dim=-1, keepdim=True).clamp_min(epsilon)
    pdp = pdp * (dims.subcarriers / dims.bs_antennas)
    return pas, pdp


def feature_memmap(path: str, sample_count: int, layout: SpectralFeatureLayout, mode: str = "r") -> np.memmap:
    return np.memmap(path, dtype=np.float32, mode=mode, shape=(sample_count, layout.total_size))


def nonzero_feature_indices(features: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Return authoritative non-outlier rows; all-zero channels map to zero features."""
    features = np.asarray(features)
    return np.flatnonzero(np.max(np.abs(features), axis=1) > epsilon)
