from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from physical_ai.data import RoundData
from physical_ai.features import SpectralFeatureLayout, spectral_targets_from_features
from physical_ai.metrics import StreamingScore, cosine_similarity_last
from physical_ai.neighbors import distance_weights, nearest_neighbors
from physical_ai.spectral import alternating_spectral_projection
from physical_ai.transductive import transductive_graph_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU differentiable PAS/PDP synthesis benchmark")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/cv_noout_split20260804.pt"))
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, nargs="+", default=[20, 80, 200])
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--output", type=Path, default=Path("artifacts/synthesis_gpu_benchmark.json"))
    return parser.parse_args()


def target_similarity(
    channel: torch.Tensor, target_pas: torch.Tensor, target_pdp: torch.Tensor
) -> tuple[float, float]:
    pas = torch.abs(torch.fft.fft(channel, dim=1, norm="ortho")).square()
    pdp = torch.abs(torch.fft.fft(channel, dim=-1, norm="ortho")).square()
    c1 = nn.functional.cosine_similarity(pas, target_pas, dim=1).mean().item()
    c2 = nn.functional.cosine_similarity(pdp, target_pdp, dim=-1).mean().item()
    return c1, c2


def differentiable_synthesis(
    initial: torch.Tensor,
    target_pas: torch.Tensor,
    target_pdp: torch.Tensor,
    steps: int,
    learning_rate: float,
    pas_weight: float = 0.5,
) -> torch.Tensor:
    channel = nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([channel], lr=learning_rate)
    desired_rms = torch.sqrt(torch.mean(torch.abs(initial).square(), dim=(1, 2, 3), keepdim=True))
    for _ in range(steps):
        pas = torch.abs(torch.fft.fft(channel, dim=1, norm="ortho")).square()
        pdp = torch.abs(torch.fft.fft(channel, dim=-1, norm="ortho")).square()
        pas_cos = nn.functional.cosine_similarity(pas, target_pas, dim=1).mean()
        pdp_cos = nn.functional.cosine_similarity(pdp, target_pdp, dim=-1).mean()
        loss = 1.0 - pas_weight * pas_cos - (1.0 - pas_weight) * pdp_cos
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            rms = torch.sqrt(torch.mean(torch.abs(channel).square(), dim=(1, 2, 3), keepdim=True))
            channel.mul_(desired_rms / rms.clamp_min(1e-12))
    return channel.detach()


@torch.inference_mode()
def separable_synthesis(
    initial: torch.Tensor,
    compact: torch.Tensor,
    layout: SpectralFeatureLayout,
    panel_delay: bool,
) -> torch.Tensor:
    batch = len(initial)
    pas = compact[:, : layout.pas_size].reshape(batch, 256, 4).clamp_min(0.0)
    pdp = compact[:, layout.pas_size :].reshape(batch, 2, 4, 192).clamp_min(0.0)
    angular_initial = torch.fft.fft(initial, dim=1, norm="ortho")
    angular_phase_seed = angular_initial.mean(dim=-1)
    angular_phase = angular_phase_seed / torch.abs(angular_phase_seed).clamp_min(1e-12)
    spatial = torch.fft.ifft(torch.sqrt(pas) * angular_phase, dim=1, norm="ortho")
    delay_initial = torch.fft.fft(initial, dim=-1, norm="ortho")
    if not panel_delay:
        delay_shape = pdp.mean(dim=1)
        delay_phase_seed = delay_initial.mean(dim=1)
        delay_phase = delay_phase_seed / torch.abs(delay_phase_seed).clamp_min(1e-12)
        waveform = torch.fft.ifft(torch.sqrt(delay_shape) * delay_phase, dim=-1, norm="ortho")
        return spatial[..., None] * waveform[:, None, :, :]
    output = torch.empty_like(initial)
    for panel in range(2):
        panel_slice = slice(panel * 128, (panel + 1) * 128)
        delay_phase_seed = delay_initial[:, panel_slice].mean(dim=1)
        delay_phase = delay_phase_seed / torch.abs(delay_phase_seed).clamp_min(1e-12)
        waveform = torch.fft.ifft(
            torch.sqrt(pdp[:, panel]) * delay_phase, dim=-1, norm="ortho"
        )
        output[:, panel_slice] = spatial[:, panel_slice, :, None] * waveform[:, None, :, :]
    return output


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device("cuda")
    data = RoundData(".")
    data.validate()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    positions = np.asarray(data.train_positions, dtype=np.float32)
    features = np.asarray(np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32)
    channels = data.train_channels
    val_idx = np.asarray(checkpoint["validation_indices"], dtype=np.int64)[: args.limit]
    train_idx = np.asarray(checkpoint["train_indices"], dtype=np.int64)
    local, distances = nearest_neighbors(positions[val_idx], positions[train_idx], 16)
    neighbors = train_idx[local]
    weights = distance_weights(distances, power=2.0).astype(np.float32)
    direct = np.einsum("qk,qkd->qd", weights, features[neighbors], optimize=True)
    graph = transductive_graph_features(
        positions[train_idx], positions[val_idx], features[train_idx], direct,
        k=8, power=2.0, alpha=0.25,
    )
    methods = ["ap20", "separable_global", "separable_panel"]
    methods += [f"opt{step}" for step in args.steps]
    scores = {name: StreamingScore(data.dims) for name in methods}
    target_scores = {name: [0.0, 0.0, 0] for name in methods}
    layout = SpectralFeatureLayout.from_dimensions(data.dims)
    for start in range(0, len(val_idx), args.batch_size):
        stop = min(start + args.batch_size, len(val_idx))
        compact = torch.from_numpy(graph[start:stop]).to(device)
        source = torch.from_numpy(
            np.array(channels[neighbors[start:stop]], dtype=np.complex64, copy=True)
        ).to(device)
        local_weights = torch.from_numpy(weights[start:stop]).to(device)
        initial = torch.sum(source * local_weights[:, :, None, None, None], dim=1)
        target_pas, target_pdp = spectral_targets_from_features(compact, data.dims)
        target = torch.from_numpy(
            np.array(channels[val_idx[start:stop]], dtype=np.complex64, copy=True)
        )
        candidates: dict[str, torch.Tensor] = {}
        candidates["ap20"] = torch.stack([
            alternating_spectral_projection(
                initial[i], target_pas[i], target_pdp[i], iterations=20,
                relaxation=0.5, final_constraint="pdp",
            )
            for i in range(stop - start)
        ])
        candidates["separable_global"] = separable_synthesis(initial, compact, layout, False)
        candidates["separable_panel"] = separable_synthesis(initial, compact, layout, True)
        longest = max(args.steps)
        optimized = candidates["ap20"]
        elapsed = 0
        for step in sorted(args.steps):
            optimized = differentiable_synthesis(
                optimized, target_pas, target_pdp, step - elapsed, args.learning_rate
            )
            candidates[f"opt{step}"] = optimized
            elapsed = step
        for name, prediction in candidates.items():
            c1, c2 = target_similarity(prediction, target_pas, target_pdp)
            target_scores[name][0] += c1 * (stop - start)
            target_scores[name][1] += c2 * (stop - start)
            target_scores[name][2] += stop - start
            scores[name].update((prediction.cpu() * 1e-7), target)
        print(f"processed {stop}/{len(val_idx)}", flush=True)
    result = {}
    for name in methods:
        actual = scores[name].compute()
        count = target_scores[name][2]
        actual["target_c1"] = target_scores[name][0] / count
        actual["target_c2"] = target_scores[name][1] / count
        result[name] = actual
    print(json.dumps(result, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
