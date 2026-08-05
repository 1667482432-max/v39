from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from physical_ai.features import nonzero_feature_indices
from physical_ai.grouped_model import GroupedPhysicalKernel
from experiments.search_spatial_kernels_gpu import metric_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train grouped Physical-AI attention kernel")
    parser.add_argument("--kind", choices=("pas", "pdp"), required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--split-seed", type=int, default=20260804)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("artifacts/grouped_kernel.pt"))
    return parser.parse_args()


def feature_score(prediction: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "pas":
        prediction = prediction.reshape(-1, 256, 4)
        target = target.reshape(-1, 256, 4)
        return torch.nn.functional.cosine_similarity(prediction, target, dim=1).mean()
    prediction = prediction.reshape(-1, 2, 4, 192)
    target = target.reshape(-1, 2, 4, 192)
    return torch.nn.functional.cosine_similarity(prediction, target, dim=-1).mean()


@torch.inference_mode()
def evaluate(
    model: GroupedPhysicalKernel,
    query_idx: np.ndarray,
    neighbors: np.ndarray,
    distances: np.ndarray,
    positions: torch.Tensor,
    contexts: torch.Tensor,
    features: torch.Tensor,
    kind: str,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    for start in range(0, len(query_idx), batch_size):
        stop = min(start + batch_size, len(query_idx))
        query = torch.from_numpy(query_idx[start:stop]).long().to(positions.device)
        neighbor = torch.from_numpy(neighbors[start:stop]).long().to(positions.device)
        distance = torch.from_numpy(distances[start:stop]).to(positions.device)
        prediction, _ = model(
            positions[query], contexts[query], positions[neighbor], contexts[neighbor],
            distance, features[neighbor],
        )
        total += feature_score(prediction, features[query], kind).item() * (stop - start)
    return total / len(query_idx)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    all_positions = np.load("Round1_Train_Pos.npy").astype(np.float32)
    all_contexts = np.load("artifacts/map_context.npz")["train"].astype(np.float32)
    all_features = np.asarray(np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32)
    valid = nonzero_feature_indices(all_features)
    positions_np = all_positions[valid]
    contexts_np = all_contexts[valid]
    if args.kind == "pas":
        features_np = all_features[valid, :1024]
        metric_name = "xy_ctx-all_s4"
    else:
        features_np = all_features[valid, 1024:]
        metric_name = "xy_ctx-patch_s4"
    embedding = metric_embeddings(positions_np, contexts_np)[metric_name]
    permutation = np.random.default_rng(args.split_seed).permutation(len(valid))
    val_idx = np.sort(permutation[: args.validation_size])
    train_idx = np.sort(permutation[args.validation_size :])
    tree = cKDTree(embedding[train_idx])
    train_distance, train_local = tree.query(embedding[train_idx], k=args.neighbors + 1, workers=-1)
    train_neighbors = train_idx[train_local[:, 1:]]
    train_distance = train_distance[:, 1:].astype(np.float32)
    val_distance, val_local = tree.query(embedding[val_idx], k=args.neighbors, workers=-1)
    val_neighbors = train_idx[val_local]
    val_distance = val_distance.astype(np.float32)
    device = torch.device("cuda")
    positions = torch.from_numpy(positions_np).to(device)
    contexts = torch.from_numpy(contexts_np).to(device)
    features = torch.from_numpy(features_np.copy()).to(device)
    model = GroupedPhysicalKernel(
        contexts[train_idx].mean(0), contexts[train_idx].std(0), args.kind, groups=args.groups
    ).to(device)
    baseline = evaluate(
        model, val_idx, val_neighbors, val_distance, positions, contexts, features,
        args.kind, args.batch_size,
    )
    print(f"kind={args.kind} device=cuda baseline={baseline:.9f}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    lookup = np.empty(len(valid), dtype=np.int64)
    lookup[train_idx] = np.arange(len(train_idx))
    queries = train_idx.copy()
    best_score = baseline
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(queries)
        train_total = 0.0
        for start in range(0, len(queries), args.batch_size):
            query_np = queries[start : start + args.batch_size]
            rows = lookup[query_np]
            neighbor_np = train_neighbors[rows]
            query = torch.from_numpy(query_np).long().to(device)
            neighbor = torch.from_numpy(neighbor_np).long().to(device)
            distance = torch.from_numpy(train_distance[rows]).to(device)
            prediction, residual = model(
                positions[query], contexts[query], positions[neighbor], contexts[neighbor],
                distance, features[neighbor],
            )
            score = feature_score(prediction, features[query], args.kind)
            loss = 1.0 - score + args.regularization * residual.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_total += score.item() * len(query_np)
        scheduler.step()
        validation = evaluate(
            model, val_idx, val_neighbors, val_distance, positions, contexts, features,
            args.kind, args.batch_size,
        )
        print(
            f"epoch={epoch:03d} train={train_total / len(train_idx):.9f} "
            f"val={validation:.9f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True,
        )
        if validation > best_score:
            best_score = validation
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            print(f"early_stop epoch={epoch}", flush=True)
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "kind": args.kind,
            "groups": args.groups,
            "metric_name": metric_name,
            "validation_score": best_score,
            "baseline_score": baseline,
            "best_epoch": best_epoch,
            "validation_indices": val_idx,
            "train_indices": train_idx,
            "valid_global_indices": valid,
            "neighbors": args.neighbors,
            "seed": args.seed,
        },
        args.output,
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {"kind": args.kind, "baseline": baseline, "best": best_score, "best_epoch": best_epoch},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved={args.output} best={best_score:.9f} epoch={best_epoch}")


if __name__ == "__main__":
    main()
