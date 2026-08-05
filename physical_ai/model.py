from __future__ import annotations

import torch
from torch import nn


class MapConditionedKernel(nn.Module):
    """Learn score-aligned neighbor weights from geometry and point-cloud context."""

    def __init__(
        self,
        position_mean: torch.Tensor,
        position_std: torch.Tensor,
        context_mean: torch.Tensor,
        context_std: torch.Tensor,
        ue_antennas: int = 4,
        bs_polarizations: int = 2,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.ue_antennas = ue_antennas
        self.bs_polarizations = bs_polarizations
        self.register_buffer("position_mean", position_mean.float())
        self.register_buffer("position_std", position_std.float().clamp_min(1e-4))
        self.register_buffer("context_mean", context_mean.float())
        self.register_buffer("context_std", context_std.float().clamp_min(1e-4))
        context_input = context_mean.numel() + position_mean.numel()
        self.context_encoder = nn.Sequential(
            nn.Linear(context_input, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
        )
        pair_input = 6 + 3 * (hidden // 2)
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_input, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.pas_residual = nn.Linear(hidden, ue_antennas)
        self.pdp_residual = nn.Linear(hidden, bs_polarizations * ue_antennas)
        nn.init.zeros_(self.pas_residual.weight)
        nn.init.zeros_(self.pas_residual.bias)
        nn.init.zeros_(self.pdp_residual.weight)
        nn.init.zeros_(self.pdp_residual.bias)
        self.pas_power = nn.Parameter(torch.full((ue_antennas,), 2.0))
        self.pdp_power = nn.Parameter(
            torch.full((bs_polarizations, ue_antennas), 2.0)
        )

    def _context(self, position: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        position = (position - self.position_mean) / self.position_std
        context = (context - self.context_mean) / self.context_std
        return self.context_encoder(torch.cat((position, context), dim=-1))

    def forward(
        self,
        query_position: torch.Tensor,
        query_context: torch.Tensor,
        neighbor_position: torch.Tensor,
        neighbor_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, neighbors, _ = neighbor_position.shape
        query_embedding = self._context(query_position, query_context)
        neighbor_embedding = self._context(
            neighbor_position.reshape(batch * neighbors, -1),
            neighbor_context.reshape(batch * neighbors, -1),
        ).reshape(batch, neighbors, -1)
        query_expanded = query_embedding[:, None, :].expand_as(neighbor_embedding)
        delta = (neighbor_position - query_position[:, None, :]) / self.position_std
        distance = torch.linalg.vector_norm(
            neighbor_position[..., :2] - query_position[:, None, :2], dim=-1
        ).clamp_min(1e-3)
        unit = (
            neighbor_position[..., :2] - query_position[:, None, :2]
        ) / distance[..., None]
        relative = torch.cat(
            (delta, unit, torch.log1p(distance)[..., None]), dim=-1
        )
        pair = torch.cat(
            (
                relative,
                query_expanded,
                neighbor_embedding,
                torch.abs(query_expanded - neighbor_embedding),
            ),
            dim=-1,
        )
        encoded = self.pair_encoder(pair)
        log_distance = torch.log(distance)
        pas_scores = (
            -log_distance[..., None] * self.pas_power.clamp(0.25, 6.0)
            + self.pas_residual(encoded)
        )
        pdp_scores = (
            -log_distance[..., None, None]
            * self.pdp_power.clamp(0.25, 6.0)
            + self.pdp_residual(encoded).reshape(
                batch, neighbors, self.bs_polarizations, self.ue_antennas
            )
        )
        return torch.softmax(pas_scores, dim=1), torch.softmax(pdp_scores, dim=1)


def interpolate_features(
    neighbor_features: torch.Tensor,
    pas_weights: torch.Tensor,
    pdp_weights: torch.Tensor,
    pas_size: int,
    bs_antennas: int = 256,
    ue_antennas: int = 4,
    bs_polarizations: int = 2,
    subcarriers: int = 192,
) -> torch.Tensor:
    batch, neighbors, _ = neighbor_features.shape
    source_pas = neighbor_features[..., :pas_size].reshape(
        batch, neighbors, bs_antennas, ue_antennas
    )
    source_pdp = neighbor_features[..., pas_size:].reshape(
        batch, neighbors, bs_polarizations, ue_antennas, subcarriers
    )
    prediction_pas = torch.einsum("bkn,bkmn->bmn", pas_weights, source_pas)
    prediction_pdp = torch.einsum("bkpn,bkpns->bpns", pdp_weights, source_pdp)
    return torch.cat((prediction_pas.flatten(1), prediction_pdp.flatten(1)), dim=1)
