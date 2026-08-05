from __future__ import annotations

import torch
from torch import nn


class DelaywiseNeighborAttention(nn.Module):
    """Learn query-dependent neighbor weights independently for every delay bin."""

    def __init__(self, pair_features: int, hidden: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(pair_features + 5, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.correction_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        pair: torch.Tensor,
        coherence: torch.Tensor,
        log_energy: torch.Tensor,
        log_spatial_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, neighbors, delays = coherence.shape
        delay = torch.arange(delays, device=coherence.device, dtype=coherence.dtype)
        angle = 2.0 * torch.pi * delay / delays
        circular = torch.minimum(delay, delays - delay) / (delays / 2.0)
        delay_feature = torch.stack((torch.sin(angle), torch.cos(angle), circular), dim=-1)
        delay_feature = delay_feature[None, None].expand(batch, neighbors, -1, -1)
        expanded_pair = pair[:, :, None].expand(-1, -1, delays, -1)
        dynamic = torch.stack((coherence, log_energy), dim=-1)
        correction = self.network(
            torch.cat((expanded_pair, dynamic, delay_feature), dim=-1)
        ).squeeze(-1)
        logits = log_spatial_weight[:, :, None] + self.correction_scale * correction
        return torch.softmax(logits, dim=1), correction


def angle_delay_coefficients(channels: torch.Tensor) -> torch.Tensor:
    """Transform (..., M, N, S) channels to the joint angle-delay domain."""
    return torch.fft.fft(
        torch.fft.fft(channels, dim=-3, norm="ortho"), dim=-1, norm="ortho"
    )


def observable_delay_statistics(coefficients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return standardized anchor coherence and log-energy per neighbor/delay."""
    energy = torch.sum(torch.abs(coefficients).square(), dim=(-3, -2)).clamp_min(1e-30)
    anchor = coefficients[:, :1]
    coherence = torch.abs(torch.sum(torch.conj(coefficients) * anchor, dim=(-3, -2)))
    coherence /= torch.sqrt(energy * energy[:, :1]).clamp_min(1e-30)
    coherence = (coherence - coherence.mean(1, keepdim=True)) / coherence.std(
        1, keepdim=True
    ).clamp_min(1e-4)
    log_energy = torch.log(energy)
    log_energy = (log_energy - log_energy.mean(1, keepdim=True)) / log_energy.std(
        1, keepdim=True
    ).clamp_min(1e-4)
    return coherence, log_energy


def phase_aligned_idw(
    channels: torch.Tensor, spatial_weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Globally phase-align neighbors to the nearest one and form their IDW centroid."""
    anchor = channels[:, :1]
    cross = torch.sum(torch.conj(channels) * anchor, dim=(-3, -2, -1), keepdim=True)
    phase = cross / torch.abs(cross).clamp_min(1e-30)
    aligned = channels * phase
    centroid = torch.sum(aligned * spatial_weight[:, :, None, None, None], dim=1)
    return aligned, centroid


def reconstruct_from_attention(
    coefficients: torch.Tensor,
    weights: torch.Tensor,
    centroid: torch.Tensor,
    hidw_blend: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse magnitudes, use nearest-neighbor phase, invert, and norm/phase-align to IDW."""
    amplitude = torch.sum(weights[:, :, None, None, :] * torch.abs(coefficients), dim=1)
    anchor_phase = coefficients[:, 0] / torch.abs(coefficients[:, 0]).clamp_min(1e-30)
    fused_coefficients = amplitude * anchor_phase
    prediction = torch.fft.ifft(
        torch.fft.ifft(fused_coefficients, dim=-1, norm="ortho"), dim=-3, norm="ortho"
    )
    axes = (-3, -2, -1)
    cross = torch.sum(torch.conj(prediction) * centroid, dim=axes, keepdim=True)
    prediction_energy = torch.sum(
        torch.abs(prediction).square(), dim=axes, keepdim=True
    ).clamp_min(1e-30)
    centroid_energy = torch.sum(torch.abs(centroid).square(), dim=axes, keepdim=True)
    prediction = prediction * torch.sqrt(centroid_energy / prediction_energy) * (
        cross / torch.abs(cross).clamp_min(1e-30)
    )
    return (1.0 - hidw_blend) * prediction + hidw_blend * centroid, fused_coefficients
