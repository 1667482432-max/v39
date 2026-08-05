from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

from physical_ai.data import RoundData
from physical_ai.expert_gate import ExpertDisagreementGate
from physical_ai.features import nonzero_feature_indices
from experiments.search_kriging_ensemble_gpu import CONFIGS, Config, component_score, predict_config
from experiments.search_local_kriging_gpu import covariance
from experiments.search_spatial_kernels_gpu import metric_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an observable expert-disagreement gating model")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/cv_noout_split20260804.pt"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--correction-scale", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("artifacts/disagreement_gating.json"))
    return parser.parse_args()


def predict_loo(
    config: Config,
    embedding: np.ndarray,
    indices: np.ndarray,
    features: torch.Tensor,
) -> torch.Tensor:
    distance_np, local = cKDTree(embedding[indices]).query(
        embedding[indices], k=config.k + 1, workers=-1
    )
    distance_np = distance_np[:, 1:]
    neighbor_np = indices[local[:, 1:]]
    neighbor_embedding = embedding[neighbor_np]
    pair_np = np.linalg.norm(
        neighbor_embedding[:, :, None, :] - neighbor_embedding[:, None, :, :], axis=-1
    ).astype(np.float32)
    device = features.device
    pair = torch.from_numpy(pair_np).to(device)
    distance = torch.from_numpy(distance_np.astype(np.float32)).to(device)
    bandwidth = distance[:, -1:, None] * config.scale
    cov_nn = covariance(pair, bandwidth, config.kind)
    cov_q = covariance(distance, bandwidth[:, :, 0], config.kind)
    q, k = distance.shape
    system = torch.zeros((q, k + 1, k + 1), device=device)
    system[:, :k, :k] = cov_nn + torch.eye(k, device=device) * config.nugget
    system[:, :k, k] = 1.0
    system[:, k, :k] = 1.0
    right = torch.cat((cov_q, torch.ones((q, 1), device=device)), dim=1)
    weight = torch.linalg.solve(system, right)[:, :k]
    if config.positive:
        weight = weight.clamp_min(0.0)
        weight /= weight.sum(1, keepdim=True).clamp_min(1e-8)
    neighbor = torch.from_numpy(neighbor_np.astype(np.int64)).to(device)
    return torch.einsum("qk,qkd->qd", weight, features[neighbor]).clamp_min(0.0)


def train_component(
    train_expert: torch.Tensor,
    train_target: torch.Tensor,
    val_expert: torch.Tensor,
    val_target: torch.Tensor,
    condition_train: torch.Tensor,
    condition_val: torch.Tensor,
    base_weight: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    groups = train_expert.shape[1]
    model = ExpertDisagreementGate(condition_train.shape[1], groups, train_expert.shape[2]).to(train_expert.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    base_logits = torch.log(base_weight.clamp_min(1e-7))[None]
    best_score = -1.0
    best_state = None
    for step in range(1, args.steps + 1):
        model.train()
        correction = model(condition_train, train_expert)
        weight = torch.softmax(base_logits + args.correction_scale * correction, dim=-1)
        prediction = torch.einsum("qgc,qgcl->qgl", weight, train_expert)
        score = torch.nn.functional.cosine_similarity(prediction, train_target, dim=-1).mean()
        penalty = 1e-4 * correction.square().mean()
        loss = 1.0 - score + penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == args.steps:
            model.eval()
            with torch.inference_mode():
                val_correction = model(condition_val, val_expert)
                val_weight = torch.softmax(
                    base_logits + args.correction_scale * val_correction, dim=-1
                )
                val_prediction = torch.einsum("qgc,qgcl->qgl", val_weight, val_expert)
                val_score = torch.nn.functional.cosine_similarity(
                    val_prediction, val_target, dim=-1
                ).mean().item()
            if val_score > best_score:
                best_score = val_score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"step={step} train={score.item():.6f} val={val_score:.6f}", flush=True)
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        correction = model(condition_val, val_expert)
        weight = torch.softmax(base_logits + args.correction_scale * correction, dim=-1)
        prediction = torch.einsum("qgc,qgcl->qgl", weight, val_expert)
        baseline = torch.einsum("gc,qgcl->qgl", base_weight, val_expert)
        metrics = {
            "baseline": torch.nn.functional.cosine_similarity(
                baseline, val_target, dim=-1
            ).mean().item(),
            "gated": torch.nn.functional.cosine_similarity(
                prediction, val_target, dim=-1
            ).mean().item(),
        }
    return prediction, metrics


def main() -> None:
    args = parse_args()
    torch.manual_seed(20260805)
    device = torch.device("cuda")
    data = RoundData(".")
    data.validate()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    suffix = args.checkpoint.stem.removeprefix("cv_noout_split")
    all_features = np.asarray(np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32)
    valid = nonzero_feature_indices(all_features)
    inverse = np.full(len(all_features), -1, dtype=np.int64)
    inverse[valid] = np.arange(len(valid))
    positions = np.asarray(data.train_positions, dtype=np.float32)[valid]
    contexts = np.load("artifacts/map_context.npz")["train"].astype(np.float32)[valid]
    features = torch.from_numpy(all_features[valid]).to(device)
    train_idx = inverse[np.asarray(checkpoint["train_indices"], dtype=np.int64)]
    val_idx = inverse[np.asarray(checkpoint["validation_indices"], dtype=np.int64)]
    embeddings = metric_embeddings(positions, contexts)
    train_bank = torch.stack([
        predict_loo(config, embeddings[config.metric], train_idx, features) for config in CONFIGS
    ])
    print("generated train expert bank", flush=True)
    val_bank = torch.stack([
        predict_config(config, embeddings[config.metric], train_idx, val_idx, features) for config in CONFIGS
    ])
    print("generated validation expert bank", flush=True)
    condition_np = np.concatenate((positions[:, :2], contexts[:, :7]), axis=1)
    mean, std = condition_np[train_idx].mean(0), condition_np[train_idx].std(0).clip(1e-4)
    condition = torch.from_numpy(((condition_np - mean) / std).astype(np.float32)).to(device)
    meta = json.loads(Path("artifacts/groupwise_kriging_v2.json").read_text(encoding="utf-8"))
    fold_meta = next(row for row in meta["loo"] if row["fold"] == suffix)
    base_weight = torch.tensor(fold_meta["weights"], dtype=torch.float32, device=device)

    train_pas = train_bank[:, :, :1024].reshape(len(CONFIGS), -1, 256, 4).permute(1, 3, 0, 2)
    val_pas = val_bank[:, :, :1024].reshape(len(CONFIGS), -1, 256, 4).permute(1, 3, 0, 2)
    target_pas_train = features[train_idx, :1024].reshape(-1, 256, 4).permute(0, 2, 1)
    target_pas_val = features[val_idx, :1024].reshape(-1, 256, 4).permute(0, 2, 1)
    _, pas_metrics = train_component(
        train_pas, target_pas_train, val_pas, target_pas_val,
        condition[train_idx], condition[val_idx], base_weight[:4], args,
    )

    train_pdp = train_bank[:, :, 1024:].reshape(len(CONFIGS), -1, 8, 192).permute(1, 2, 0, 3)
    val_pdp = val_bank[:, :, 1024:].reshape(len(CONFIGS), -1, 8, 192).permute(1, 2, 0, 3)
    target_pdp_train = features[train_idx, 1024:].reshape(-1, 8, 192)
    target_pdp_val = features[val_idx, 1024:].reshape(-1, 8, 192)
    _, pdp_metrics = train_component(
        train_pdp, target_pdp_train, val_pdp, target_pdp_val,
        condition[train_idx], condition[val_idx], base_weight[4:], args,
    )
    result = {"fold": suffix, "pas": pas_metrics, "pdp": pdp_metrics}
    print(json.dumps(result, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
