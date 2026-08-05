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
    p = argparse.ArgumentParser(description="Fine carrier search with steered ordinary kriging")
    p.add_argument("--checkpoint", type=Path, default=Path("artifacts/cv_noout_split20260804.pt"))
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--minimum", type=float, default=138.0)
    p.add_argument("--maximum", type=float, default=142.5)
    p.add_argument("--step", type=float, default=0.025)
    p.add_argument("--output", type=Path, default=Path("artifacts/kriging_wavenumber.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = RoundData("."); data.validate()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    features = np.asarray(np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32)
    valid = nonzero_feature_indices(features)
    inverse = np.full(len(features), -1, dtype=np.int64); inverse[valid] = np.arange(len(valid))
    positions = np.asarray(data.train_positions, dtype=np.float64)[valid]
    val_global = np.asarray(checkpoint["validation_indices"], dtype=np.int64)[:args.limit]
    train_global = np.asarray(checkpoint["train_indices"], dtype=np.int64)
    val_idx, train_idx = inverse[val_global], inverse[train_global]
    local, distance = nearest_neighbors(positions[val_idx], positions[train_idx], 16)
    neighbor = train_idx[local]
    neighbor_xy = positions[neighbor, :2]
    pair = np.linalg.norm(neighbor_xy[:, :, None] - neighbor_xy[:, None, :], axis=-1)
    bandwidth = np.maximum(distance[:, -1] * 0.75, 1e-6)
    cnn = np.exp(-pair / bandwidth[:, None, None])
    cq = np.exp(-distance / bandwidth[:, None])
    system = np.zeros((len(val_idx), 17, 17)); system[:, :16, :16] = cnn + np.eye(16)[None] * 0.1
    system[:, :16, 16] = 1; system[:, 16, :16] = 1
    weight = np.linalg.solve(system, np.c_[cq, np.ones(len(val_idx))][..., None])[..., 0][:, :16]
    weight = np.maximum(weight, 0); weight /= weight.sum(1, keepdims=True)
    bs = np.asarray(data.dims.bs_position); offset = positions - bs
    radius = np.linalg.norm(offset, axis=1); direction = offset / radius[:, None]
    dr = radius[neighbor] - radius[val_idx, None]
    du = direction[neighbor] - direction[val_idx, None]
    device = torch.device("cuda")
    frequency = torch.arange(data.dims.subcarriers, device=device, dtype=torch.float32)
    frequency -= (data.dims.subcarriers - 1) / 2
    h = torch.arange(data.dims.bs_h, device=device, dtype=torch.float32) - (data.dims.bs_h - 1) / 2
    v = torch.arange(data.dims.bs_v, device=device, dtype=torch.float32) - (data.dims.bs_v - 1) / 2
    hg = h[:, None].expand(data.dims.bs_h, data.dims.bs_v).reshape(-1).repeat(data.dims.bs_polarizations)
    vg = v[None].expand(data.dims.bs_h, data.dims.bs_v).reshape(-1).repeat(data.dims.bs_polarizations)
    cross = np.empty((len(val_idx), 16), np.complex128)
    gram = np.empty((len(val_idx), 16, 16), np.complex128)
    target_energy = np.empty(len(val_idx))
    channels = data.train_channels
    for q in range(len(val_idx)):
        source = torch.from_numpy(np.array(channels[valid[neighbor[q]]], copy=True)).to(device)
        target = torch.from_numpy(np.array(channels[val_global[q]], copy=True)).to(device)
        delta_r = torch.from_numpy(dr[q].astype(np.float32)).to(device)
        delta_u = torch.from_numpy(du[q].astype(np.float32)).to(device)
        slope_phase = torch.exp(1j * delta_r[:, None] * 0.0006 * frequency)
        steering = torch.exp(1j * ((-1.75*delta_u[:,0,None]-2.5*delta_u[:,1,None])*hg + 26*delta_u[:,2,None]*vg))
        adjusted = source * slope_phase[:, None, None, :] * steering[:, :, None, None]
        flat = adjusted.reshape(16, -1); target = target.reshape(-1)
        cross[q] = (torch.conj(flat) @ target).cpu().numpy()
        gram[q] = (torch.conj(flat) @ flat.T).cpu().numpy()
        target_energy[q] = torch.sum(torch.abs(target).square(), dtype=torch.float64).item()
        if (q+1)%25==0: print(f"correlations {q+1}/{len(val_idx)}", flush=True)
    waves = np.arange(args.minimum, args.maximum + args.step/2, args.step)
    rho = []
    for wave in waves:
        coefficient = weight * np.exp(1j * wave * dr)
        c = np.einsum("qk,qk->", np.conj(coefficient), cross)
        ep = np.einsum("qk,qkl,ql->", np.conj(coefficient), gram, coefficient).real
        rho.append(abs(c)**2/(ep*target_energy.sum()))
    rho = np.asarray(rho); best = int(np.argmax(rho))
    result = {"best_wavenumber":float(waves[best]),"best_nmse":float(1-rho[best]),"at_140_25":float(1-rho[np.argmin(abs(waves-140.25))])}
    print(json.dumps(result,indent=2)); args.output.write_text(json.dumps(result,indent=2))


if __name__ == "__main__": main()
