from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import RoundDimensions


def cosine_similarity_last(
    prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-30
) -> torch.Tensor:
    numerator = torch.sum(prediction * target, dim=-1)
    prediction_norm = torch.linalg.vector_norm(prediction, dim=-1)
    target_norm = torch.linalg.vector_norm(target, dim=-1)
    denominator = prediction_norm * target_norm
    both_zero = (prediction_norm <= epsilon) & (target_norm <= epsilon)
    similarity = numerator / denominator.clamp_min(epsilon)
    return torch.where(both_zero, torch.ones_like(similarity), similarity)


def pas_spectrum(channel: torch.Tensor, dims: RoundDimensions) -> torch.Tensor:
    if channel.shape[1:] != dims.channel_shape:
        raise ValueError("Channel shape does not match M/N/S")
    angular = torch.fft.fft(channel, dim=1, norm="ortho")
    return (torch.abs(angular) ** 2).permute(0, 2, 3, 1)


def pdp_spectrum(channel: torch.Tensor) -> torch.Tensor:
    impulse = torch.fft.fft(channel, dim=-1, norm="ortho")
    return torch.abs(impulse) ** 2


def score_components(
    prediction: torch.Tensor, target: torch.Tensor, dims: RoundDimensions
) -> dict[str, torch.Tensor]:
    pred_pas, target_pas = pas_spectrum(prediction, dims), pas_spectrum(target, dims)
    pred_pdp, target_pdp = pdp_spectrum(prediction), pdp_spectrum(target)
    c1 = cosine_similarity_last(pred_pas, target_pas).mean()
    c2 = cosine_similarity_last(pred_pdp, target_pdp).mean()
    error = torch.sum(torch.abs(prediction - target) ** 2)
    energy = torch.sum(torch.abs(target) ** 2)
    c3 = error / energy.clamp_min(1e-20)
    w1, w2, w3 = dims.score_weights
    score = w1 * c1 + w2 * c2 + w3 / (1.0 + c3)
    return {"c1_pas": c1, "c2_pdp": c2, "c3_nmse": c3, "score": score}


@dataclass
class StreamingScore:
    dims: RoundDimensions
    pas_sum: float = 0.0
    pas_count: int = 0
    pdp_sum: float = 0.0
    pdp_count: int = 0
    error: float = 0.0
    energy: float = 0.0
    prediction_energy: float = 0.0
    real_cross: float = 0.0

    @torch.inference_mode()
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.to(torch.complex64)
        target = target.to(torch.complex64)
        pp, tp = pas_spectrum(prediction, self.dims), pas_spectrum(target, self.dims)
        pd, td = pdp_spectrum(prediction), pdp_spectrum(target)
        pas_cos = cosine_similarity_last(pp, tp)
        pdp_cos = cosine_similarity_last(pd, td)
        self.pas_sum += pas_cos.sum(dtype=torch.float64).item()
        self.pas_count += pas_cos.numel()
        self.pdp_sum += pdp_cos.sum(dtype=torch.float64).item()
        self.pdp_count += pdp_cos.numel()
        self.error += torch.sum(torch.abs(prediction - target) ** 2, dtype=torch.float64).item()
        self.energy += torch.sum(torch.abs(target) ** 2, dtype=torch.float64).item()
        self.prediction_energy += torch.sum(
            torch.abs(prediction) ** 2, dtype=torch.float64
        ).item()
        self.real_cross += torch.sum(
            torch.real(torch.conj(prediction) * target), dtype=torch.float64
        ).item()

    def compute(self) -> dict[str, float]:
        c1 = self.pas_sum / self.pas_count
        c2 = self.pdp_sum / self.pdp_count
        c3 = self.error / max(self.energy, 1e-30)
        w1, w2, w3 = self.dims.score_weights
        optimal_scale = max(0.0, self.real_cross / max(self.prediction_energy, 1e-30))
        optimal_nmse = (
            self.energy
            + optimal_scale * optimal_scale * self.prediction_energy
            - 2.0 * optimal_scale * self.real_cross
        ) / max(self.energy, 1e-30)
        return {
            "c1_pas": c1,
            "c2_pdp": c2,
            "c3_nmse": c3,
            "score": w1*c1+w2*c2+w3/(1+c3),
            "optimal_nonnegative_scale": optimal_scale,
            "score_at_optimal_scale": w1*c1+w2*c2+w3/(1+optimal_nmse),
            "nmse_at_optimal_scale": optimal_nmse,
        }
