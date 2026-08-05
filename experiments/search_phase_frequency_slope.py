from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from physical_ai.data import RoundData
from physical_ai.features import nonzero_feature_indices
from physical_ai.neighbors import nearest_neighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search frequency-dependent radial phase slope")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/cv_noout_split20260804.pt"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--wavenumber", type=float, default=140.25)
    parser.add_argument("--max-slope", type=float, default=0.02)
    parser.add_argument("--slope-step", type=float, default=0.00025)
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase_frequency_slope.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = RoundData(".")
    data.validate()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    features = np.asarray(np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32)
    valid = nonzero_feature_indices(features)
    inverse = np.full(len(features), -1, dtype=np.int64)
    inverse[valid] = np.arange(len(valid))
    positions = np.asarray(data.train_positions, dtype=np.float64)[valid]
    val_global = np.asarray(checkpoint["validation_indices"], dtype=np.int64)[: args.limit]
    train_global = np.asarray(checkpoint["train_indices"], dtype=np.int64)
    val_idx, train_idx = inverse[val_global], inverse[train_global]
    local, distance = nearest_neighbors(positions[val_idx], positions[train_idx], 12)
    neighbors = train_idx[local]
    weights = (distance + 2.0) ** -4.0
    weights /= weights.sum(axis=1, keepdims=True)
    radius = np.linalg.norm(positions - np.array([50.0, 0.0, 25.0]), axis=1)
    radial_delta = radius[neighbors] - radius[val_idx, None]

    device = torch.device("cuda")
    channels = data.train_channels
    q_count, k_count, subcarriers = len(val_idx), 12, channels.shape[-1]
    cross = torch.empty((q_count, k_count, subcarriers), dtype=torch.complex64, device=device)
    gram = torch.empty((q_count, k_count, k_count, subcarriers), dtype=torch.complex64, device=device)
    target_energy = torch.empty(q_count, dtype=torch.float64, device=device)
    for q in range(q_count):
        source = torch.from_numpy(np.array(channels[valid[neighbors[q]]], copy=True)).to(device)
        target = torch.from_numpy(np.array(channels[val_global[q]], copy=True)).to(device)
        cross[q] = torch.einsum("knms,nms->ks", torch.conj(source), target)
        gram[q] = torch.einsum("knms,lnms->kls", torch.conj(source), source)
        target_energy[q] = torch.sum(torch.abs(target).square(), dtype=torch.float64)
        if (q + 1) % 25 == 0:
            print(f"correlations {q + 1}/{q_count}", flush=True)

    slopes = np.arange(-args.max_slope, args.max_slope + args.slope_step * 0.5, args.slope_step)
    frequency_index = torch.arange(subcarriers, device=device, dtype=torch.float32)
    frequency_index -= (subcarriers - 1) / 2.0
    delta = torch.from_numpy(radial_delta.astype(np.float32)).to(device)
    base_weight = torch.from_numpy(weights.astype(np.float32)).to(device)
    per_query_cross = np.empty((len(slopes), q_count), dtype=np.complex128)
    per_query_energy = np.empty((len(slopes), q_count), dtype=np.float64)
    for si, slope in enumerate(slopes):
        phase_k = args.wavenumber + float(slope) * frequency_index
        coefficient = base_weight[:, :, None] * torch.exp(1j * delta[:, :, None] * phase_k)
        cq = torch.einsum("qks,qks->q", torch.conj(coefficient), cross)
        eq = torch.einsum(
            "qks,qkls,qls->q", torch.conj(coefficient), gram, coefficient
        ).real
        per_query_cross[si] = cq.cpu().numpy().astype(np.complex128)
        per_query_energy[si] = eq.cpu().numpy().astype(np.float64)
        if (si + 1) % 20 == 0:
            print(f"slopes {si + 1}/{len(slopes)}", flush=True)

    target_energy_np = target_energy.cpu().numpy()
    total_cross = per_query_cross.sum(axis=1)
    total_prediction_energy = per_query_energy.sum(axis=1)
    total_target_energy = target_energy_np.sum()
    rho2 = np.abs(total_cross) ** 2 / (total_prediction_energy * total_target_energy)
    best = int(np.argmax(rho2))

    rng = np.random.default_rng(20260804)
    subfolds = np.array_split(rng.permutation(q_count), 5)
    heldout = []
    for fold, test_rows in enumerate(subfolds):
        train_rows = np.concatenate([rows for j, rows in enumerate(subfolds) if j != fold])
        ctrain = per_query_cross[:, train_rows].sum(axis=1)
        eptrain = per_query_energy[:, train_rows].sum(axis=1)
        ettrain = target_energy_np[train_rows].sum()
        quality = np.abs(ctrain) ** 2 / (eptrain * ettrain)
        selected = int(np.argmax(quality))
        scale = ctrain[selected] / eptrain[selected]
        ctest = per_query_cross[selected, test_rows].sum()
        eptest = per_query_energy[selected, test_rows].sum()
        ettest = target_energy_np[test_rows].sum()
        nmse = (ettest + abs(scale) ** 2 * eptest - 2 * np.real(np.conj(scale) * ctest)) / ettest
        heldout.append({"fold": fold, "slope": float(slopes[selected]), "nmse": float(nmse)})

    zero = int(np.argmin(np.abs(slopes)))
    result = {
        "best_slope": float(slopes[best]),
        "best_nmse": float(1.0 - rho2[best]),
        "zero_slope_nmse": float(1.0 - rho2[zero]),
        "heldout": heldout,
        "heldout_mean_nmse": float(np.mean([row["nmse"] for row in heldout])),
    }
    print(json.dumps(result, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
