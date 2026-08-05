from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from physical_ai.data import RoundData
from physical_ai.features import SpectralFeatureLayout, nonzero_feature_indices
from physical_ai.model import MapConditionedKernel, interpolate_features
from physical_ai.neighbors import nearest_neighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the map-conditioned Physical AI kernel")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--features", type=Path, default=Path("artifacts/spectral_features.npy"))
    parser.add_argument("--context", type=Path, default=Path("artifacts/map_context.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.pt"))
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--split-seed", type=int, default=20260804)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device. 'auto' uses CUDA when it is available.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def score_loss(prediction: torch.Tensor, target: torch.Tensor, layout: SpectralFeatureLayout) -> torch.Tensor:
    pas_p = prediction[:, : layout.pas_size].reshape(-1, 256, 4)
    pas_t = target[:, : layout.pas_size].reshape(-1, 256, 4)
    pdp_p = prediction[:, layout.pas_size :].reshape(-1, 2, 4, 192)
    pdp_t = target[:, layout.pas_size :].reshape(-1, 2, 4, 192)
    pas_cos = nn.functional.cosine_similarity(pas_p, pas_t, dim=1).mean()
    pdp_cos = nn.functional.cosine_similarity(pdp_p, pdp_t, dim=-1).mean()
    return 1.0 - 0.5 * (pas_cos + pdp_cos)


@torch.inference_mode()
def evaluate(
    model: MapConditionedKernel,
    queries: np.ndarray,
    neighbor_indices: np.ndarray,
    positions: torch.Tensor,
    contexts: torch.Tensor,
    features: torch.Tensor,
    layout: SpectralFeatureLayout,
    batch_size: int,
) -> float:
    losses = []
    model.eval()
    device = positions.device
    for start in range(0, len(queries), batch_size):
        query = torch.from_numpy(queries[start : start + batch_size]).long().to(device)
        neighbor = torch.from_numpy(neighbor_indices[start : start + batch_size]).long().to(device)
        pw, dw = model(positions[query], contexts[query], positions[neighbor], contexts[neighbor])
        prediction = interpolate_features(features[neighbor], pw, dw, layout.pas_size)
        losses.append(score_loss(prediction, features[query], layout).item() * len(query))
    return 1.0 - sum(losses) / len(queries)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    print(f"device={device}", flush=True)
    data = RoundData(args.root)
    data.validate()
    positions_np = np.asarray(data.train_positions, dtype=np.float32)
    contexts_np = np.load(args.context)["train"].astype(np.float32)
    features_np = np.asarray(np.load(args.features, mmap_mode="r"), dtype=np.float32)
    positions = torch.from_numpy(positions_np).to(device)
    contexts = torch.from_numpy(contexts_np).to(device)
    features = torch.from_numpy(features_np.copy()).to(device)
    layout = SpectralFeatureLayout.from_dimensions(data.dims)
    valid_indices = nonzero_feature_indices(features_np)
    if len(valid_indices) != len(features_np):
        print(
            f"excluding_zero_channel_outliers={len(features_np) - len(valid_indices)} "
            f"valid_samples={len(valid_indices)}",
            flush=True,
        )
    split_rng = np.random.default_rng(args.split_seed)
    train_rng = np.random.default_rng(args.seed)
    permutation = valid_indices[split_rng.permutation(len(valid_indices))]
    val_idx = np.sort(permutation[: args.validation_size])
    train_idx = np.sort(permutation[args.validation_size :])
    train_local, _ = nearest_neighbors(positions_np[train_idx], positions_np[train_idx], args.neighbors + 1)
    train_neighbors = train_idx[train_local[:, 1:]]
    if len(val_idx):
        val_local, _ = nearest_neighbors(positions_np[val_idx], positions_np[train_idx], args.neighbors)
        val_neighbors = train_idx[val_local]
    else:
        val_neighbors = np.empty((0, args.neighbors), dtype=np.int64)
    model = MapConditionedKernel(
        positions[train_idx].mean(0),
        positions[train_idx].std(0),
        contexts[train_idx].mean(0),
        contexts[train_idx].std(0),
    ).to(device)
    if len(val_idx):
        baseline_validation = evaluate(
            model, val_idx, val_neighbors, positions, contexts, features, layout, args.batch_size
        )
        print(f"fixed_kernel_validation={baseline_validation:.6f}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_score, best_state = -float("inf"), None
    train_queries = train_idx.copy()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_rng.shuffle(train_queries)
        # Neighbor rows are keyed by sorted train_idx; recover them through a lookup.
        row_lookup = np.empty(len(positions_np), dtype=np.int64)
        row_lookup[train_idx] = np.arange(len(train_idx))
        running = 0.0
        for start in range(0, len(train_queries), args.batch_size):
            query_np = train_queries[start : start + args.batch_size]
            neighbor_np = train_neighbors[row_lookup[query_np]]
            query = torch.from_numpy(query_np).long().to(device)
            neighbor = torch.from_numpy(neighbor_np).long().to(device)
            pas_weights, pdp_weights = model(
                positions[query], contexts[query], positions[neighbor], contexts[neighbor]
            )
            prediction = interpolate_features(features[neighbor], pas_weights, pdp_weights, layout.pas_size)
            loss = score_loss(prediction, features[query], layout)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running += loss.item() * len(query)
        scheduler.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            train_score = 1.0 - running / len(train_idx)
            if len(val_idx):
                validation_score = evaluate(
                    model, val_idx, val_neighbors, positions, contexts, features, layout, args.batch_size
                )
                print(
                    f"epoch={epoch:03d} train={train_score:.6f} val={validation_score:.6f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )
                if validation_score > best_score:
                    best_score = validation_score
                    best_state = {
                        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                    }
            else:
                print(
                    f"epoch={epoch:03d} train={train_score:.6f} lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )
                best_score = train_score
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "validation_score": best_score,
            "neighbors": args.neighbors,
            "seed": args.seed,
            "validation_indices": val_idx,
            "train_indices": train_idx,
            "context_dim": contexts.shape[1],
            "excluded_indices": np.setdiff1d(np.arange(len(features_np)), valid_indices),
        },
        args.output,
    )
    args.output.with_suffix(".json").write_text(
        json.dumps({"validation_score": best_score, "epochs": args.epochs}, indent=2), encoding="utf-8"
    )
    print(f"saved {args.output}; best_validation_score={best_score:.6f}")


if __name__ == "__main__":
    main()
