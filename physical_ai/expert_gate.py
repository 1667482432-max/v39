from __future__ import annotations

import numpy as np
import torch
from torch import nn


def expert_gate_condition(
    positions: np.ndarray, contexts: np.ndarray, mode: str = "basic"
) -> np.ndarray:
    """Return observable Physical-AI conditions for expert selection.

    ``basic`` preserves the v34 gate input.  ``advanced`` adds compact,
    physically interpretable point-cloud descriptors without feeding the gate
    the full 209-column raster encoding: center-ray clearance/material, the
    16/32 m terminal environment, and smoothed local fields.
    """
    position = np.asarray(positions, dtype=np.float32)
    context = np.asarray(contexts, dtype=np.float32)
    basic = np.concatenate((position[:, :2], context[:, :7]), axis=1)
    if mode == "basic":
        return basic
    if mode != "advanced":
        raise ValueError(f"Unsupported expert-gate condition mode: {mode}")
    if context.shape[1] < 362:
        raise ValueError(
            f"Advanced expert gating needs 362 context columns, got {context.shape[1]}"
        )
    advanced = context[:, 153:]
    return np.concatenate(
        (
            basic,
            advanced[:, 18:24],   # center-ray clearance/transition summary
            advanced[:, 54:58],   # center-ray wall material and orientation
            advanced[:, 91:105],  # terminal environment at 16/32 m
            advanced[:, 201:209], # smoothed height/wall fields at four scales
        ),
        axis=1,
    ).astype(np.float32, copy=False)


class ExpertDisagreementGate(nn.Module):
    """Observable residual gate driven by map context and expert disagreement."""

    def __init__(self, condition_dim: int, groups: int, experts: int) -> None:
        super().__init__()
        self.group_embedding = nn.Embedding(groups, 12)
        self.network = nn.Sequential(
            nn.Linear(condition_dim + experts * experts + experts + 12, 96),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(96, 64),
            nn.GELU(),
            nn.Linear(64, experts),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, condition: torch.Tensor, expert: torch.Tensor) -> torch.Tensor:
        unit = expert / torch.linalg.vector_norm(
            expert, dim=-1, keepdim=True
        ).clamp_min(1e-12)
        gram = torch.einsum("qgcl,qgdl->qgcd", unit, unit).flatten(2)
        norm = torch.log(torch.linalg.vector_norm(expert, dim=-1).clamp_min(1e-12))
        norm = norm - norm.mean(-1, keepdim=True)
        queries, groups = expert.shape[:2]
        group = self.group_embedding(
            torch.arange(groups, device=expert.device)
        )[None].expand(queries, -1, -1)
        context = condition[:, None].expand(-1, groups, -1)
        return self.network(torch.cat((context, gram, norm, group), dim=-1))
