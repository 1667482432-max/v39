from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from physical_ai.data import RoundData
from physical_ai.delay_attention import (
    DelaywiseNeighborAttention,
    angle_delay_coefficients,
    observable_delay_statistics,
    phase_aligned_idw,
    reconstruct_from_attention,
)
from physical_ai.features import nonzero_feature_indices
from physical_ai.metrics import cosine_similarity_last, pas_spectrum, pdp_spectrum
from physical_ai.neighbors import nearest_neighbors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train leave-one-out delay-wise neighbor attention")
    p.add_argument("--checkpoint", type=Path, default=Path("artifacts/cv_noout_split20260804.pt"))
    p.add_argument("--neighbors", type=int, default=8)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--output", type=Path, default=Path("artifacts/delaywise_attention_model.pt"))
    return p.parse_args()


def make_pair_features(
    positions: np.ndarray,
    contexts: np.ndarray,
    query: np.ndarray,
    neighbors: np.ndarray,
    distances: np.ndarray,
    position_mean: np.ndarray,
    position_std: np.ndarray,
    context_mean: np.ndarray,
    context_std: np.ndarray,
) -> np.ndarray:
    qpos = (positions[query, :2] - position_mean) / position_std
    delta = (positions[neighbors, :2] - positions[query, None, :2]) / position_std
    unit = positions[neighbors, :2] - positions[query, None, :2]
    unit /= np.maximum(distances[..., None], 1e-6)
    qctx = (contexts[query, :7] - context_mean) / context_std
    cdelta = (contexts[neighbors, :7] - contexts[query, None, :7]) / context_std
    rank = np.arange(neighbors.shape[1], dtype=np.float32)[None, :, None] / max(neighbors.shape[1] - 1, 1)
    return np.concatenate(
        (
            np.broadcast_to(qpos[:, None], (*neighbors.shape, 2)),
            delta,
            unit,
            np.log1p(distances)[..., None],
            np.broadcast_to(rank, (*neighbors.shape, 1)),
            np.broadcast_to(qctx[:, None], (*neighbors.shape, 7)),
            cdelta,
        ),
        axis=-1,
    ).astype(np.float32)


def steering_geometry(positions: np.ndarray, query: np.ndarray, neighbors: np.ndarray, bs: np.ndarray):
    radius = np.linalg.norm(positions - bs, axis=1)
    direction = (positions - bs) / radius[:, None]
    return radius[neighbors] - radius[query, None], direction[neighbors] - direction[query, None]


def main() -> None:
    args = parse_args()
    torch.manual_seed(20260805)
    rng = np.random.default_rng(20260805)
    device = torch.device("cuda")
    data = RoundData(".")
    data.validate()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    features = np.asarray(np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32)
    valid = nonzero_feature_indices(features)
    inverse = np.full(len(features), -1, dtype=np.int64)
    inverse[valid] = np.arange(len(valid))
    positions = np.asarray(data.train_positions, dtype=np.float32)[valid]
    contexts = np.load("artifacts/map_context.npz")["train"].astype(np.float32)[valid]
    train_idx = inverse[np.asarray(checkpoint["train_indices"], dtype=np.int64)]
    val_idx = inverse[np.asarray(checkpoint["validation_indices"], dtype=np.int64)]
    train_local, train_distance = nearest_neighbors(positions[train_idx, :2], positions[train_idx, :2], args.neighbors + 1)
    train_neighbor = train_idx[train_local[:, 1:]]
    train_distance = train_distance[:, 1:]
    val_local, val_distance = nearest_neighbors(positions[val_idx, :2], positions[train_idx, :2], args.neighbors)
    val_neighbor = train_idx[val_local]
    position_mean = positions[train_idx, :2].mean(0)
    position_std = positions[train_idx, :2].std(0).clip(1e-4)
    context_mean = contexts[train_idx, :7].mean(0)
    context_std = contexts[train_idx, :7].std(0).clip(1e-4)
    train_pair = make_pair_features(positions, contexts, train_idx, train_neighbor, train_distance, position_mean, position_std, context_mean, context_std)
    val_pair = make_pair_features(positions, contexts, val_idx, val_neighbor, val_distance, position_mean, position_std, context_mean, context_std)
    bs = np.asarray(data.dims.bs_position, dtype=np.float32)
    train_radial, train_direction = steering_geometry(positions, train_idx, train_neighbor, bs)
    val_radial, val_direction = steering_geometry(positions, val_idx, val_neighbor, bs)
    frequency = torch.arange(data.dims.subcarriers, device=device, dtype=torch.float32) - (data.dims.subcarriers - 1) / 2.0
    h = torch.arange(data.dims.bs_h, device=device, dtype=torch.float32) - (data.dims.bs_h - 1) / 2.0
    v = torch.arange(data.dims.bs_v, device=device, dtype=torch.float32) - (data.dims.bs_v - 1) / 2.0
    h_grid = h[:, None].expand(data.dims.bs_h, data.dims.bs_v).reshape(-1).repeat(data.dims.bs_polarizations)
    v_grid = v[None, :].expand(data.dims.bs_h, data.dims.bs_v).reshape(-1).repeat(data.dims.bs_polarizations)

    def prepare(query, neighbor, distance, pair, radial, direction, rows):
        source = torch.from_numpy(np.array(data.train_channels[valid[neighbor[rows]]], dtype=np.complex64, copy=True)).to(device)
        target = torch.from_numpy(np.array(data.train_channels[valid[query[rows]]], dtype=np.complex64, copy=True)).to(device)
        rd = torch.from_numpy(radial[rows]).to(device)
        dd = torch.from_numpy(direction[rows]).to(device)
        phase = torch.exp(1j * rd[:, :, None] * (140.25 + 0.0006 * frequency))
        steering = torch.exp(1j * ((-1.75 * dd[:, :, 0, None] - 2.5 * dd[:, :, 1, None]) * h_grid + 26.0 * dd[:, :, 2, None] * v_grid))
        source = source * steering[:, :, :, None, None] * phase[:, :, None, None, :]
        spatial = torch.from_numpy(np.maximum(distance[rows], 1e-6).astype(np.float32)).to(device).pow(-2.0)
        spatial /= spatial.sum(1, keepdim=True)
        aligned, centroid = phase_aligned_idw(source, spatial)
        coefficient = angle_delay_coefficients(aligned)
        coherence, log_energy = observable_delay_statistics(coefficient)
        return source, target, coefficient, coherence, log_energy, spatial, centroid, torch.from_numpy(pair[rows]).to(device)

    model = DelaywiseNeighborAttention(train_pair.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)

    def spectral_loss(coefficient, target, weights, correction):
        amplitude = torch.sum(weights[:, :, None, None, :] * torch.abs(coefficient), dim=1)
        anchor_phase = coefficient[:, 0] / torch.abs(coefficient[:, 0]).clamp_min(1e-30)
        fused = amplitude * anchor_phase
        target_coefficient = angle_delay_coefficients(target)
        amp_cos = nn.functional.cosine_similarity(
            amplitude.permute(0, 3, 1, 2).flatten(2),
            torch.abs(target_coefficient).permute(0, 3, 1, 2).flatten(2),
            dim=-1,
        ).mean()
        pred_pas = torch.abs(torch.fft.ifft(fused, dim=-1, norm="ortho")).square()
        true_pas = torch.abs(torch.fft.ifft(target_coefficient, dim=-1, norm="ortho")).square()
        pas_cos = nn.functional.cosine_similarity(pred_pas, true_pas, dim=1).mean()
        pred_pdp = torch.abs(torch.fft.ifft(fused, dim=1, norm="ortho")).square()
        true_pdp = torch.abs(torch.fft.ifft(target_coefficient, dim=1, norm="ortho")).square()
        pdp_cos = nn.functional.cosine_similarity(pred_pdp, true_pdp, dim=-1).mean()
        return 1.0 - (0.2 * amp_cos + 0.4 * pas_cos + 0.4 * pdp_cos) + 1e-5 * correction.square().mean()

    best_score, best_state, history = -1.0, None, []
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train_idx))
        model.train()
        train_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            rows = order[start : start + args.batch_size]
            _, target, coefficient, coherence, log_energy, spatial, _, pair = prepare(train_idx, train_neighbor, train_distance, train_pair, train_radial, train_direction, rows)
            weight, correction = model(pair, coherence, log_energy, torch.log(spatial.clamp_min(1e-20)))
            loss = spectral_loss(coefficient, target, weight, correction)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_loss += loss.item() * len(rows)
        model.eval()
        totals = {"pas": 0.0, "pdp": 0.0, "count": 0, "cross": 0j, "pred": 0.0, "target": 0.0}
        with torch.inference_mode():
            for start in range(0, len(val_idx), args.batch_size):
                rows = np.arange(start, min(start + args.batch_size, len(val_idx)))
                _, target, coefficient, coherence, log_energy, spatial, centroid, pair = prepare(val_idx, val_neighbor, val_distance, val_pair, val_radial, val_direction, rows)
                weight, _ = model(pair, coherence, log_energy, torch.log(spatial.clamp_min(1e-20)))
                prediction, _ = reconstruct_from_attention(coefficient, weight, centroid, 0.1)
                batch = len(rows)
                totals["pas"] += cosine_similarity_last(pas_spectrum(prediction, data.dims), pas_spectrum(target, data.dims)).mean().item() * batch
                totals["pdp"] += cosine_similarity_last(pdp_spectrum(prediction), pdp_spectrum(target)).mean().item() * batch
                totals["count"] += batch
                totals["cross"] += torch.sum(torch.conj(prediction) * target).item()
                totals["pred"] += torch.sum(torch.abs(prediction).square(), dtype=torch.float64).item()
                totals["target"] += torch.sum(torch.abs(target).square(), dtype=torch.float64).item()
        c1, c2 = totals["pas"] / totals["count"], totals["pdp"] / totals["count"]
        nmse = 1.0 - abs(totals["cross"]) ** 2 / (totals["pred"] * totals["target"])
        score = 0.4 * c1 + 0.4 * c2 + 0.2 / (1.0 + nmse)
        row = {"epoch": epoch, "train_loss": train_loss / len(train_idx), "pas": c1, "pdp": c2, "nmse": nmse, "score": score}
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "pair_features": train_pair.shape[-1], "position_mean": position_mean, "position_std": position_std, "context_mean": context_mean, "context_std": context_std, "neighbors": args.neighbors, "best_score": best_score, "history": history}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps({"best_score": best_score, "history": history}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
