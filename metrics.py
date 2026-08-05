from __future__ import annotations

import torch

from physical_ai.data import RoundDimensions


def cosine_similarity_last(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-30) -> torch.Tensor:
    numerator = torch.sum(prediction * target, dim=-1)
    prediction_norm = torch.linalg.vector_norm(prediction, dim=-1)
    target_norm = torch.linalg.vector_norm(target, dim=-1)
    denominator = prediction_norm * target_norm
    both_zero = (prediction_norm <= epsilon) & (target_norm <= epsilon)
    similarity = numerator / denominator.clamp_min(epsilon)
    return torch.where(both_zero, torch.ones_like(similarity), similarity)


def pas_spectrum(channel: torch.Tensor, dims: RoundDimensions) -> torch.Tensor:
    if channel.shape[1:] != (dims.bs_antennas, dims.ue_antennas, dims.subcarriers):
        raise ValueError("Channel shape does not match the configured M/N/S dimensions")
    # The reference-like local convention treats the flattened BS antenna axis
    # (polarization -> H -> V) as the spatial sequence used for PAS.
    angular = torch.fft.fft(channel, dim=1, norm="ortho")
    power = torch.abs(angular) ** 2
    return power.permute(0, 2, 3, 1)


def pdp_spectrum(channel: torch.Tensor) -> torch.Tensor:
    # FFT matches the data convention used by the reproducible KNN/MLP baseline.
    impulse_response = torch.fft.fft(channel, dim=-1, norm="ortho")
    return torch.abs(impulse_response) ** 2


def channel_nmse(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-20) -> torch.Tensor:
    error = torch.sum(torch.abs(prediction - target) ** 2)
    energy = torch.sum(torch.abs(target) ** 2)
    return error / energy.clamp_min(epsilon)


def channel_nmse_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-20,
) -> torch.Tensor:
    error = torch.sum(torch.abs(prediction - target) ** 2, dim=(1, 2, 3))
    energy = torch.sum(torch.abs(target) ** 2, dim=(1, 2, 3))
    return error / energy.clamp_min(epsilon)


def channel_error_energy(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.sum(torch.abs(prediction - target) ** 2), torch.sum(torch.abs(target) ** 2)


def score_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    dims: RoundDimensions,
) -> dict[str, torch.Tensor]:
    pred_pas = pas_spectrum(prediction, dims)
    target_pas = pas_spectrum(target, dims)
    c1 = cosine_similarity_last(pred_pas, target_pas).mean()
    pred_pdp = pdp_spectrum(prediction)
    target_pdp = pdp_spectrum(target)
    c2 = cosine_similarity_last(pred_pdp, target_pdp).mean()
    c3 = channel_nmse(prediction, target)
    combined = 0.4 * c1 + 0.4 * c2 + 0.2 / (1.0 + c3)
    return {"c1_pas": c1, "c2_pdp": c2, "c3_nmse": c3, "score": combined}
