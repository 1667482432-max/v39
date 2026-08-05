from __future__ import annotations

import torch


def _mix_unit_shape_with_energy(
    spectra: torch.Tensor,
    weights: torch.Tensor,
    vector_dim: int,
    epsilon: float = 1e-20,
) -> torch.Tensor:
    """Mix normalized spectral shapes while retaining KNN-estimated L1 energy."""
    vector_norm = torch.linalg.vector_norm(spectra, dim=vector_dim, keepdim=True)
    unit = spectra / vector_norm.clamp_min(epsilon)
    view_shape = (weights.numel(),) + (1,) * (spectra.ndim - 1)
    mixed_shape = torch.sum(unit * weights.view(view_shape), dim=0)
    raw_mixture = torch.sum(spectra * weights.view(view_shape), dim=0)
    # Removing the leading neighbor axis shifts every non-negative axis by one.
    output_vector_dim = vector_dim - 1 if vector_dim >= 0 else vector_dim
    desired_energy = torch.sum(raw_mixture, dim=output_vector_dim, keepdim=True)
    return (
        mixed_shape
        / mixed_shape.sum(dim=output_vector_dim, keepdim=True).clamp_min(epsilon)
        * desired_energy
    )


@torch.inference_mode()
def knn_spectral_targets(
    neighbor_channels: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict PAS and PDP powers from local channels.

    Inputs use ``(K, M, N, S)``. The output PAS and PDP both use
    ``(M, N, S)``; their vector axes are M and S respectively.
    """
    angular = torch.fft.fft(neighbor_channels, dim=1, norm="ortho")
    delay = torch.fft.fft(neighbor_channels, dim=-1, norm="ortho")
    pas = torch.abs(angular).square()
    pdp = torch.abs(delay).square()
    return (
        _mix_unit_shape_with_energy(pas, weights, vector_dim=1),
        _mix_unit_shape_with_energy(pdp, weights, vector_dim=-1),
    )


@torch.inference_mode()
def physical_axis_denoise(
    pas: torch.Tensor,
    pdp: torch.Tensor,
    bs_polarizations: int = 2,
    bs_h: int = 16,
    bs_v: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply validation-backed invariances of the MIMO-OFDM channel powers."""
    epsilon = 1e-20
    # Normalize every scored vector first. Otherwise high-energy neighbors or
    # array elements would bias a shape-only cosine objective during smoothing.
    pas = pas / torch.linalg.vector_norm(pas, dim=0, keepdim=True).clamp_min(epsilon)
    pdp = pdp / torch.linalg.vector_norm(pdp, dim=-1, keepdim=True).clamp_min(epsilon)
    # Far-field angular support is effectively constant over the narrow OFDM band.
    pas = pas.mean(dim=-1, keepdim=True).expand_as(pas)
    # Delay power is shared by the elements of one BS polarization panel.
    shaped = pdp.reshape(bs_polarizations, bs_h, bs_v, *pdp.shape[1:])
    shaped = shaped.mean(dim=(1, 2), keepdim=True).expand_as(shaped)
    pdp = shaped.reshape_as(pdp)
    # Select harmless per-vector scales that make PAS/PDP total energies agree
    # for every UE antenna. These scales do not change either cosine metric.
    pas = pas / pas.sum(dim=0, keepdim=True).clamp_min(epsilon)
    pdp = pdp / pdp.sum(dim=-1, keepdim=True).clamp_min(epsilon)
    pdp = pdp * (pas.shape[-1] / pas.shape[0])
    return pas, pdp


def replace_fourier_magnitude(
    channel: torch.Tensor,
    target_power: torch.Tensor,
    dim: int,
    relaxation: float = 1.0,
    epsilon: float = 1e-20,
) -> torch.Tensor:
    transformed = torch.fft.fft(channel, dim=dim, norm="ortho")
    current_magnitude = torch.abs(transformed)
    target_magnitude = torch.sqrt(target_power.clamp_min(0.0))
    magnitude = current_magnitude.lerp(target_magnitude, relaxation)
    phase = transformed / current_magnitude.clamp_min(epsilon)
    return torch.fft.ifft(magnitude * phase, dim=dim, norm="ortho")


@torch.inference_mode()
def alternating_spectral_projection(
    initial_channel: torch.Tensor,
    target_pas: torch.Tensor,
    target_pdp: torch.Tensor,
    iterations: int = 5,
    relaxation: float = 1.0,
    final_constraint: str = "pdp",
) -> torch.Tensor:
    if final_constraint not in {"pas", "pdp"}:
        raise ValueError("final_constraint must be 'pas' or 'pdp'")
    channel = initial_channel
    for _ in range(iterations):
        if final_constraint == "pdp":
            channel = replace_fourier_magnitude(channel, target_pas, dim=0, relaxation=relaxation)
            channel = replace_fourier_magnitude(channel, target_pdp, dim=-1, relaxation=relaxation)
        else:
            channel = replace_fourier_magnitude(channel, target_pdp, dim=-1, relaxation=relaxation)
            channel = replace_fourier_magnitude(channel, target_pas, dim=0, relaxation=relaxation)
    return channel
