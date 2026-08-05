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
    parser = argparse.ArgumentParser(description="Search observable neighbor-spectrum consensus kernels")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/cv_noout_split20260804.pt"))
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("artifacts/neighbor_consensus.json"))
    return parser.parse_args()


def cosine_group(prediction: torch.Tensor, target: torch.Tensor, dim: int) -> float:
    return torch.nn.functional.cosine_similarity(prediction, target, dim=dim).mean().item()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    data = RoundData(".")
    data.validate()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    all_features = np.asarray(np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32)
    valid = nonzero_feature_indices(all_features)
    inverse = np.full(len(all_features), -1, dtype=np.int64)
    inverse[valid] = np.arange(len(valid))
    positions = np.asarray(data.train_positions, dtype=np.float32)[valid]
    features = torch.from_numpy(all_features[valid]).to(device)
    val_idx = inverse[np.asarray(checkpoint["validation_indices"], dtype=np.int64)]
    train_idx = inverse[np.asarray(checkpoint["train_indices"], dtype=np.int64)]
    local, distance = nearest_neighbors(positions[val_idx], positions[train_idx], args.neighbors)
    neighbor_idx = train_idx[local]
    source = features[neighbor_idx]
    target = features[val_idx]
    source_pas = source[:, :, :1024].reshape(len(val_idx), args.neighbors, 256, 4)
    source_pdp = source[:, :, 1024:].reshape(len(val_idx), args.neighbors, 2, 4, 192)
    target_pas = target[:, :1024].reshape(len(val_idx), 256, 4)
    target_pdp = target[:, 1024:].reshape(len(val_idx), 2, 4, 192)
    results: dict[str, dict[str, float]] = {}
    for k in (8, 16, 24, 32, 48, 64):
        if k > args.neighbors:
            continue
        d = torch.from_numpy(distance[:, :k].astype(np.float32)).to(device)
        pas = source_pas[:, :k]
        pdp = source_pdp[:, :k]
        pas_unit = pas / torch.linalg.vector_norm(pas, dim=2, keepdim=True).clamp_min(1e-20)
        pdp_unit = pdp / torch.linalg.vector_norm(pdp, dim=-1, keepdim=True).clamp_min(1e-20)
        pas_gram = torch.einsum("qkmu,qlmu->qukl", pas_unit, pas_unit)
        pdp_gram = torch.einsum("qkpus,qlpus->qpukl", pdp_unit, pdp_unit)
        for power in (1.0, 2.0, 3.0, 4.0):
            base = d.clamp_min(1e-3).pow(-power)
            base /= base.sum(1, keepdim=True)
            other = base[:, None, :].expand(-1, k, -1).clone()
            other *= 1.0 - torch.eye(k, device=device)[None]
            other /= other.sum(-1, keepdim=True).clamp_min(1e-20)
            pas_consensus = torch.einsum("qukl,qkl->qku", pas_gram, other)
            pdp_consensus = torch.einsum("qpukl,qkl->qkpu", pdp_gram, other)
            pas_z = (pas_consensus - pas_consensus.mean(1, keepdim=True)) / pas_consensus.std(
                1, keepdim=True
            ).clamp_min(1e-4)
            pdp_z = (pdp_consensus - pdp_consensus.mean(1, keepdim=True)) / pdp_consensus.std(
                1, keepdim=True
            ).clamp_min(1e-4)
            log_base = torch.log(base.clamp_min(1e-20))
            for strength in (-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
                pas_weight = torch.softmax(log_base[:, :, None] + strength * pas_z, dim=1)
                pdp_weight = torch.softmax(
                    log_base[:, :, None, None] + strength * pdp_z, dim=1
                )
                pred_pas = torch.einsum("qku,qkmu->qmu", pas_weight, pas)
                pred_pdp = torch.einsum("qkpu,qkpus->qpus", pdp_weight, pdp)
                c1 = cosine_group(pred_pas, target_pas, 1)
                c2 = cosine_group(pred_pdp, target_pdp, -1)
                name = f"k{k}_p{power:g}_c{strength:g}"
                results[name] = {"pas": c1, "pdp": c2, "mean": 0.5 * (c1 + c2)}
    top_pas = sorted(results.items(), key=lambda row: row[1]["pas"], reverse=True)[:20]
    top_pdp = sorted(results.items(), key=lambda row: row[1]["pdp"], reverse=True)[:20]
    print(json.dumps({"top_pas": top_pas, "top_pdp": top_pdp}, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
