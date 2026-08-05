from __future__ import annotations

import torch
from torch import nn


class GroupedPhysicalKernel(nn.Module):
    """Map-conditioned local attention with angle/delay-specific physical heads."""

    def __init__(
        self,
        context_mean: torch.Tensor,
        context_std: torch.Tensor,
        kind: str,
        groups: int = 16,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        if kind not in {"pas", "pdp"}:
            raise ValueError("kind must be 'pas' or 'pdp'")
        self.kind = kind
        self.groups = groups
        self.register_buffer("context_mean", context_mean.float())
        self.register_buffer("context_std", context_std.float().clamp_min(1e-3))
        context_dim = context_mean.numel()
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
        )
        pair_dim = 7 + 3 * (hidden // 2)
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        heads = groups * 4 if kind == "pas" else 2 * 4 * groups
        self.residual_head = nn.Linear(hidden, heads)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _encode_context(self, context: torch.Tensor) -> torch.Tensor:
        return self.context_encoder((context - self.context_mean) / self.context_std)

    def attention_logits(
        self,
        query_position: torch.Tensor,
        query_context: torch.Tensor,
        neighbor_position: torch.Tensor,
        neighbor_context: torch.Tensor,
        metric_distance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, neighbors, _ = neighbor_position.shape
        query_embedding = self._encode_context(query_context)
        neighbor_embedding = self._encode_context(
            neighbor_context.reshape(batch * neighbors, -1)
        ).reshape(batch, neighbors, -1)
        query_expanded = query_embedding[:, None].expand_as(neighbor_embedding)
        delta = neighbor_position[..., :2] - query_position[:, None, :2]
        euclidean = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-4)
        unit = delta / euclidean[..., None]
        relative = torch.cat(
            (
                delta / torch.tensor([200.0, 300.0], device=delta.device),
                unit,
                (euclidean / 20.0)[..., None],
                torch.log1p(euclidean)[..., None] / 4.0,
                torch.log1p(metric_distance.clamp_min(0.0))[..., None] / 4.0,
            ),
            dim=-1,
        )
        encoded = self.pair_encoder(
            torch.cat(
                (relative, query_expanded, neighbor_embedding, torch.abs(query_expanded - neighbor_embedding)),
                dim=-1,
            )
        )
        residual = self.residual_head(encoded)
        base = -3.0 * torch.log(metric_distance + 1.0)
        if self.kind == "pas":
            residual = residual.reshape(batch, neighbors, self.groups, 4)
            logits = base[..., None, None] + residual
        else:
            residual = residual.reshape(batch, neighbors, 2, 4, self.groups)
            logits = base[..., None, None, None] + residual
        return logits, residual

    def forward(
        self,
        query_position: torch.Tensor,
        query_context: torch.Tensor,
        neighbor_position: torch.Tensor,
        neighbor_context: torch.Tensor,
        metric_distance: torch.Tensor,
        neighbor_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, residual = self.attention_logits(
            query_position, query_context, neighbor_position, neighbor_context, metric_distance
        )
        weights = torch.softmax(logits, dim=1)
        if self.kind == "pas":
            batch, neighbors = neighbor_features.shape[:2]
            source = neighbor_features.reshape(batch, neighbors, self.groups, 256 // self.groups, 4)
            prediction = torch.einsum("bkgn,bkgln->bgln", weights, source)
        else:
            batch, neighbors = neighbor_features.shape[:2]
            source = neighbor_features.reshape(
                batch, neighbors, 2, 4, self.groups, 192 // self.groups
            )
            prediction = torch.einsum("bkpng,bkpngl->bpngl", weights, source)
        return prediction.flatten(1), residual
