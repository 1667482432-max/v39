from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.search_improved_graph_gpu import graph_matrices
from experiments.search_kriging_ensemble_gpu import CONFIGS, predict_config
from experiments.search_spatial_kernels_gpu import metric_embeddings
from physical_ai.features import nonzero_feature_indices
from physical_ai.expert_gate import ExpertDisagreementGate, expert_gate_condition


FOLDS = ("20260804", "101", "202", "303", "404")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LOFO expert-disagreement residual gating")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--correction-scale", type=float, default=0.25)
    p.add_argument("--kind", choices=("all", "pas", "pdp"), default="all")
    p.add_argument("--ensemble-size", type=int, default=1)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument(
        "--context", type=Path, default=Path("artifacts/map_context.npz")
    )
    p.add_argument(
        "--condition-mode", choices=("basic", "advanced"), default="basic"
    )
    p.add_argument(
        "--groupwise", type=Path, default=Path("artifacts/groupwise_kriging_v2.json")
    )
    p.add_argument(
        "--conditional",
        type=Path,
        default=Path("artifacts/conditional_groupwise_lowreg.json"),
    )
    p.add_argument(
        "--output", type=Path, default=Path("artifacts/disagreement_gating_cv.pt")
    )
    return p.parse_args()


def grouped_bank(bank: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "pas":
        return bank[:, :, :1024].reshape(len(CONFIGS), -1, 256, 4).permute(1, 3, 0, 2)
    return bank[:, :, 1024:].reshape(len(CONFIGS), -1, 8, 192).permute(1, 2, 0, 3)


def grouped_target(target: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "pas":
        return target[:, :1024].reshape(-1, 256, 4).permute(0, 2, 1)
    return target[:, 1024:].reshape(-1, 8, 192)


def fit_gate(
    expert: torch.Tensor,
    target: torch.Tensor,
    condition: torch.Tensor,
    base_logits: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
) -> tuple[ExpertDisagreementGate, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    mean = condition.mean(0)
    std = condition.std(0).clamp_min(1e-3)
    x = (condition - mean) / std
    model = ExpertDisagreementGate(x.shape[1], expert.shape[1], expert.shape[2]).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    for _ in range(args.steps):
        model.train()
        correction = model(x, expert)
        weight = torch.softmax(base_logits + args.correction_scale * correction, dim=-1)
        prediction = torch.einsum("qgc,qgcl->qgl", weight, expert)
        score = torch.nn.functional.cosine_similarity(prediction, target, dim=-1).mean()
        loss = 1.0 - score + 1e-4 * correction.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
    model.eval()
    return model, mean.cpu().numpy(), std.cpu().numpy()


@torch.inference_mode()
def evaluate(
    models: list[ExpertDisagreementGate],
    expert: torch.Tensor,
    target: torch.Tensor,
    condition: torch.Tensor,
    mean: np.ndarray,
    std: np.ndarray,
    base_logits: torch.Tensor,
    scale: float,
) -> tuple[float, float]:
    baseline = torch.einsum(
        "qgc,qgcl->qgl", torch.softmax(base_logits, dim=-1), expert
    )
    x = (condition - torch.from_numpy(mean).cuda()) / torch.from_numpy(std).cuda()
    correction = torch.stack([model(x, expert) for model in models]).mean(0)
    prediction = torch.einsum(
        "qgc,qgcl->qgl",
        torch.softmax(base_logits + scale * correction, dim=-1),
        expert,
    )
    return (
        torch.nn.functional.cosine_similarity(baseline, target, dim=-1).mean().item(),
        torch.nn.functional.cosine_similarity(prediction, target, dim=-1).mean().item(),
    )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    positions_all = np.load("Round1_Train_Pos.npy").astype(np.float32)
    contexts_all = np.load(args.context)["train"].astype(np.float32)
    features_all = np.asarray(
        np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32
    )
    valid = nonzero_feature_indices(features_all)
    inverse = np.full(len(features_all), -1, dtype=np.int64)
    inverse[valid] = np.arange(len(valid))
    positions, contexts = positions_all[valid], contexts_all[valid]
    raw_condition = expert_gate_condition(positions, contexts, args.condition_mode)
    raw_conditional_condition = expert_gate_condition(positions, contexts, "basic")
    features = torch.from_numpy(features_all[valid].copy()).to(device)
    embeddings = metric_embeddings(positions, contexts)
    groupwise = json.loads(args.groupwise.read_text(encoding="utf-8"))
    conditional = json.loads(args.conditional.read_text(encoding="utf-8"))
    reg_key = f'{float(conditional["final"]["regularization"]):g}'

    fold_bank: list[torch.Tensor] = []
    fold_target: list[torch.Tensor] = []
    fold_condition: list[torch.Tensor] = []
    fold_logits: dict[str, list[torch.Tensor]] = {"pas": [], "pdp": []}
    for suffix in FOLDS:
        checkpoint = torch.load(
            f"artifacts/cv_noout_split{suffix}.pt", map_location="cpu", weights_only=False
        )
        val_global = np.asarray(checkpoint["validation_indices"], dtype=np.int64)
        train_global = np.asarray(checkpoint["train_indices"], dtype=np.int64)
        query, train = inverse[val_global], inverse[train_global]
        bank = torch.stack(
            [
                predict_config(config, embeddings[config.metric], train, query, features)
                for config in CONFIGS
            ]
        )
        transition, boundary = graph_matrices(
            embeddings["xy_y0.75"], train, query, features, 24, 2.5, 0.0
        )
        bank = torch.linalg.solve(
            torch.eye(len(query), device=device) - 0.1 * transition,
            (0.9 * bank + 0.1 * boundary[None])
            .permute(1, 0, 2)
            .reshape(len(query), -1),
        ).reshape(len(query), len(CONFIGS), -1).permute(1, 0, 2)
        target = features[query]
        condition_np = raw_condition[query]
        conditional_condition_np = raw_conditional_condition[query]
        condition_tensor = torch.from_numpy(condition_np).to(device)
        fold_meta = next(row for row in groupwise["loo"] if row["fold"] == suffix)
        base = torch.tensor(fold_meta["weights"], dtype=torch.float32, device=device)
        conditional_fold = next(
            row
            for row in conditional["regularizations"][reg_key]
            if row["fold"] == suffix
        )
        normalized = torch.from_numpy(
            (
                (
                    conditional_condition_np
                    - np.asarray(conditional_fold["condition_mean"], np.float32)
                )
                / np.asarray(conditional_fold["condition_std"], np.float32)
            ).astype(np.float32)
        ).to(device)
        corrections = torch.tensor(
            conditional_fold["pas_corrections"], dtype=torch.float32, device=device
        )
        pas_logits = torch.stack(
            [
                torch.log(base[group].clamp_min(1e-7))[None]
                + normalized @ corrections[group]
                for group in range(4)
            ],
            dim=1,
        )
        pdp_logits = torch.log(base[4:].clamp_min(1e-7))[None].expand(
            len(query), -1, -1
        )
        fold_bank.append(bank)
        fold_target.append(target)
        fold_condition.append(condition_tensor)
        fold_logits["pas"].append(pas_logits)
        fold_logits["pdp"].append(pdp_logits)
        print(f"generated fold={suffix}", flush=True)

    results: dict[str, list[dict[str, float | str]]] = {"pas": [], "pdp": []}
    final_payload: dict[str, object] = {}
    fold_payload: dict[str, dict[str, object]] = {"pas": {}, "pdp": {}}
    kinds = ("pas", "pdp") if args.kind == "all" else (args.kind,)
    for kind in kinds:
        experts = [grouped_bank(bank, kind) for bank in fold_bank]
        targets = [grouped_target(target, kind) for target in fold_target]
        for heldout, suffix in enumerate(FOLDS):
            training = [i for i in range(len(FOLDS)) if i != heldout]
            models = []
            for member in range(args.ensemble_size):
                model, mean, std = fit_gate(
                    torch.cat([experts[i] for i in training]),
                    torch.cat([targets[i] for i in training]),
                    torch.cat([fold_condition[i] for i in training]),
                    torch.cat([fold_logits[kind][i] for i in training]),
                    args,
                    20260805 + heldout + args.seed_offset + 1000 * member,
                )
                models.append(model)
            baseline, gated = evaluate(
                models,
                experts[heldout],
                targets[heldout],
                fold_condition[heldout],
                mean,
                std,
                fold_logits[kind][heldout],
                args.correction_scale,
            )
            row = {"fold": suffix, "baseline": baseline, "gated": gated}
            results[kind].append(row)
            fold_payload[kind][suffix] = {
                "state_dicts": [
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                    for model in models
                ],
                "condition_mean": mean,
                "condition_std": std,
                "groups": experts[heldout].shape[1],
                "experts": experts[heldout].shape[2],
            }
            print(f"kind={kind} {row}", flush=True)
        final_models = []
        for member in range(args.ensemble_size):
            final_model, mean, std = fit_gate(
                torch.cat(experts),
                torch.cat(targets),
                torch.cat(fold_condition),
                torch.cat(fold_logits[kind]),
                args,
                (20260825 if kind == "pas" else 20260826)
                + args.seed_offset
                + 1000 * member,
            )
            final_models.append(final_model)
        final_payload[kind] = {
            "state_dicts": [
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
                for model in final_models
            ],
            "condition_mean": mean,
            "condition_std": std,
            "groups": experts[0].shape[1],
            "experts": experts[0].shape[2],
        }

    summary = {
        kind: {
            "baseline": float(np.mean([row["baseline"] for row in rows])),
            "gated": float(np.mean([row["gated"] for row in rows])),
        }
        for kind, rows in results.items()
        if rows
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **final_payload,
            "fold_models": fold_payload,
            "correction_scale": args.correction_scale,
            "condition_mode": args.condition_mode,
            "context": str(args.context),
            "results": results,
            "summary": summary,
        },
        args.output,
    )
    args.output.with_suffix(".json").write_text(
        json.dumps({"results": results, "summary": summary}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
