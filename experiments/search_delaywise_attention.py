from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from physical_ai.data import RoundData
from physical_ai.features import nonzero_feature_indices
from physical_ai.metrics import cosine_similarity_last, pas_spectrum, pdp_spectrum
from physical_ai.neighbors import nearest_neighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delay-wise neighbor attention reconstruction")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/cv_noout_split20260804.pt"))
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--neighbors", type=int, default=12)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--softening", type=float, default=0.0)
    parser.add_argument("--coherence-gammas", type=float, nargs="+", default=(0.0, 1.0, 2.0, 4.0))
    parser.add_argument("--energy-gammas", type=float, nargs="+", default=(0.0, 0.5))
    parser.add_argument("--fusions", nargs="+", choices=("magnitude", "power", "complex"), default=("magnitude", "power", "complex"))
    parser.add_argument("--alignments", nargs="+", choices=("none", "norm", "ls"), default=("norm", "ls"))
    parser.add_argument("--blends", type=float, nargs="+", default=(0.0, 0.1, 0.2, 0.4))
    parser.add_argument("--steered", action="store_true")
    parser.add_argument(
        "--angle-transforms", nargs="+", choices=("flat", "hv"), default=("flat",)
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/delaywise_attention.json"))
    return parser.parse_args()


def align_to_reference(prediction: torch.Tensor, reference: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return prediction
    axes = tuple(range(1, prediction.ndim))
    cross = torch.sum(torch.conj(prediction) * reference, dim=axes, keepdim=True)
    prediction_energy = torch.sum(torch.abs(prediction).square(), dim=axes, keepdim=True).clamp_min(1e-30)
    if mode == "ls":
        return prediction * (cross / prediction_energy)
    reference_energy = torch.sum(torch.abs(reference).square(), dim=axes, keepdim=True)
    scale = torch.sqrt(reference_energy / prediction_energy)
    phase = cross / torch.abs(cross).clamp_min(1e-30)
    return prediction * scale * phase


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    data = RoundData(".")
    data.validate()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    feature_bank = np.asarray(np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32)
    valid = nonzero_feature_indices(feature_bank)
    inverse = np.full(len(feature_bank), -1, dtype=np.int64)
    inverse[valid] = np.arange(len(valid))
    positions = np.asarray(data.train_positions, dtype=np.float64)[valid]
    val_global = np.asarray(checkpoint["validation_indices"], dtype=np.int64)[: args.limit]
    train_global = np.asarray(checkpoint["train_indices"], dtype=np.int64)
    val_idx, train_idx = inverse[val_global], inverse[train_global]
    local, distance = nearest_neighbors(positions[val_idx, :2], positions[train_idx, :2], args.neighbors)
    neighbor_idx = train_idx[local]
    base_weight = np.maximum(distance + args.softening, 1e-6) ** (-args.power)
    base_weight /= base_weight.sum(axis=1, keepdims=True)

    if args.steered:
        bs = np.asarray(data.dims.bs_position, dtype=np.float64)
        radius = np.linalg.norm(positions - bs, axis=1)
        direction = (positions - bs) / radius[:, None]
        radial_delta = radius[neighbor_idx] - radius[val_idx, None]
        direction_delta = direction[neighbor_idx] - direction[val_idx, None, :]
        frequency = torch.arange(data.dims.subcarriers, device=device, dtype=torch.float32)
        frequency -= (data.dims.subcarriers - 1) / 2.0
        h = torch.arange(data.dims.bs_h, device=device, dtype=torch.float32)
        h -= (data.dims.bs_h - 1) / 2.0
        v = torch.arange(data.dims.bs_v, device=device, dtype=torch.float32)
        v -= (data.dims.bs_v - 1) / 2.0
        h_grid = h[:, None].expand(data.dims.bs_h, data.dims.bs_v).reshape(-1).repeat(data.dims.bs_polarizations)
        v_grid = v[None, :].expand(data.dims.bs_h, data.dims.bs_v).reshape(-1).repeat(data.dims.bs_polarizations)

    totals = defaultdict(lambda: {"pas": 0.0, "pdp": 0.0, "count": 0, "cross": 0j, "pred_energy": 0.0, "target_energy": 0.0})

    def accumulate(name: str, prediction: torch.Tensor, target: torch.Tensor) -> None:
        batch = len(prediction)
        true_pas = pas_spectrum(target, data.dims)
        true_pdp = pdp_spectrum(target)
        item = totals[name]
        item["pas"] += cosine_similarity_last(pas_spectrum(prediction, data.dims), true_pas).mean().item() * batch
        item["pdp"] += cosine_similarity_last(pdp_spectrum(prediction), true_pdp).mean().item() * batch
        item["count"] += batch
        item["cross"] += torch.sum(torch.conj(prediction) * target).item()
        item["pred_energy"] += torch.sum(torch.abs(prediction).square(), dtype=torch.float64).item()
        item["target_energy"] += torch.sum(torch.abs(target).square(), dtype=torch.float64).item()

    channels = data.train_channels
    for start in range(0, len(val_idx), args.batch_size):
        stop = min(start + args.batch_size, len(val_idx))
        source = torch.from_numpy(np.array(channels[valid[neighbor_idx[start:stop]]], dtype=np.complex64, copy=True)).to(device)
        target = torch.from_numpy(np.array(channels[val_global[start:stop]], dtype=np.complex64, copy=True)).to(device)
        if args.steered:
            radial = torch.from_numpy(radial_delta[start:stop].astype(np.float32)).to(device)
            angular = torch.from_numpy(direction_delta[start:stop].astype(np.float32)).to(device)
            radial_phase = torch.exp(1j * radial[:, :, None] * (140.25 + 0.0006 * frequency))
            steering = torch.exp(1j * ((-1.75 * angular[:, :, 0, None] - 2.5 * angular[:, :, 1, None]) * h_grid + 26.0 * angular[:, :, 2, None] * v_grid))
            source = source * steering[:, :, :, None, None] * radial_phase[:, :, None, None, :]

        spatial_weight = torch.from_numpy(base_weight[start:stop].astype(np.float32)).to(device)
        anchor = source[:, :1]
        channel_cross = torch.sum(torch.conj(source) * anchor, dim=(2, 3, 4), keepdim=True)
        phase_align = channel_cross / torch.abs(channel_cross).clamp_min(1e-30)
        aligned_source = source * phase_align
        hidw = torch.sum(aligned_source * spatial_weight[:, :, None, None, None], dim=1)
        accumulate("nearest", source[:, 0], target)
        accumulate("hidw", hidw, target)

        for angle_transform in args.angle_transforms:
            if angle_transform == "flat":
                coefficient = torch.fft.fft(
                    torch.fft.fft(aligned_source, dim=2, norm="ortho"),
                    dim=-1,
                    norm="ortho",
                )
                reduction_dims = (2, 3)
            else:
                layout = aligned_source.reshape(
                    stop - start,
                    args.neighbors,
                    data.dims.bs_polarizations,
                    data.dims.bs_h,
                    data.dims.bs_v,
                    data.dims.ue_antennas,
                    data.dims.subcarriers,
                )
                coefficient = torch.fft.fft(
                    torch.fft.fft(
                        torch.fft.fft(layout, dim=3, norm="ortho"),
                        dim=4,
                        norm="ortho",
                    ),
                    dim=-1,
                    norm="ortho",
                )
                reduction_dims = (2, 3, 4, 5)
            coefficient_energy = torch.sum(
                torch.abs(coefficient).square(), dim=reduction_dims
            ).clamp_min(1e-30)
            anchor_coefficient = coefficient[:, :1]
            coherence = torch.abs(
                torch.sum(
                    torch.conj(coefficient) * anchor_coefficient,
                    dim=reduction_dims,
                )
            )
            coherence /= torch.sqrt(
                coefficient_energy * coefficient_energy[:, :1]
            ).clamp_min(1e-30)
            coherence_z = (coherence - coherence.mean(1, keepdim=True)) / coherence.std(
                1, keepdim=True
            ).clamp_min(1e-4)
            log_energy = torch.log(coefficient_energy)
            energy_z = (log_energy - log_energy.mean(1, keepdim=True)) / log_energy.std(
                1, keepdim=True
            ).clamp_min(1e-4)
            anchor_phase = anchor_coefficient[:, 0] / torch.abs(
                anchor_coefficient[:, 0]
            ).clamp_min(1e-30)
            log_spatial = torch.log(spatial_weight.clamp_min(1e-20))[:, :, None]

            for coherence_gamma in args.coherence_gammas:
                for energy_gamma in args.energy_gammas:
                    delay_weight = torch.softmax(
                        log_spatial
                        + coherence_gamma * coherence_z
                        + energy_gamma * energy_z,
                        dim=1,
                    )
                    expanded = delay_weight[(...,) + (None,) * (coefficient.ndim - 3) + (slice(None),)]
                    for fusion in args.fusions:
                        if fusion == "magnitude":
                            amplitude = torch.sum(
                                expanded * torch.abs(coefficient), dim=1
                            )
                        elif fusion == "power":
                            amplitude = torch.sqrt(
                                torch.sum(
                                    expanded * torch.abs(coefficient).square(), dim=1
                                ).clamp_min(0.0)
                            )
                        else:
                            amplitude = torch.abs(
                                torch.sum(expanded * coefficient, dim=1)
                            )
                        fused = amplitude * anchor_phase
                        if angle_transform == "flat":
                            prediction = torch.fft.ifft(
                                torch.fft.ifft(fused, dim=-1, norm="ortho"),
                                dim=1,
                                norm="ortho",
                            )
                        else:
                            prediction = torch.fft.ifft(
                                torch.fft.ifft(
                                    torch.fft.ifft(fused, dim=-1, norm="ortho"),
                                    dim=3,
                                    norm="ortho",
                                ),
                                dim=2,
                                norm="ortho",
                            ).reshape(stop - start, *data.dims.channel_shape)
                        for alignment in args.alignments:
                            aligned = align_to_reference(prediction, hidw, alignment)
                            for blend in args.blends:
                                candidate = (1.0 - blend) * aligned + blend * hidw
                                name = (
                                    f"{angle_transform}_c{coherence_gamma:g}_e{energy_gamma:g}_"
                                    f"{fusion}_{alignment}_b{blend:g}"
                                )
                                accumulate(name, candidate, target)
        print(f"processed {stop}/{len(val_idx)}", flush=True)

    results = {}
    for name, item in totals.items():
        c1 = item["pas"] / item["count"]
        c2 = item["pdp"] / item["count"]
        scale = item["cross"] / item["pred_energy"]
        nmse = 1.0 - abs(item["cross"]) ** 2 / max(item["pred_energy"] * item["target_energy"], 1e-30)
        results[name] = {"c1_pas": c1, "c2_pdp": c2, "optimal_complex_scale": [scale.real, scale.imag], "c3_nmse": nmse, "score": 0.4 * c1 + 0.4 * c2 + 0.2 / (1.0 + nmse)}
    top = sorted(results.items(), key=lambda row: row[1]["score"], reverse=True)
    summary = {"settings": vars(args) | {"checkpoint": str(args.checkpoint), "output": str(args.output)}, "top": top[:30], "baselines": {key: results[key] for key in ("nearest", "hidw")}}
    print(json.dumps(summary, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
