from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.search_spatial_kernels_gpu import metric_embeddings
from physical_ai.features import nonzero_feature_indices
from physical_ai.neighbors import nearest_neighbors


FOLDS = ("20260804", "101", "202", "303", "404")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search local polynomial spectral regression")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-neighbors", type=int, default=48)
    parser.add_argument("--output", type=Path, default=Path("artifacts/local_polynomial_spectral.json"))
    return parser.parse_args()


def local_linear_weights(
    query_xy: np.ndarray,
    neighbor_xy: np.ndarray,
    distance: np.ndarray,
    power: float,
    softening: float,
    ridge: float,
) -> np.ndarray:
    delta = neighbor_xy - query_xy[:, None]
    design = np.concatenate((np.ones((*delta.shape[:2], 1)), delta), axis=2)
    radial = (distance + softening + 1e-6) ** (-power)
    normal = np.einsum("qki,qk,qkj->qij", design, radial, design)
    normal[:, 1:, 1:] += np.eye(2)[None] * ridge
    right = np.zeros((len(query_xy), 3), dtype=np.float64)
    right[:, 0] = 1.0
    coefficient = np.linalg.solve(normal, right[..., None])[..., 0]
    weight = radial * np.einsum("qki,qi->qk", design, coefficient)
    return weight.astype(np.float32)


def group_cosines(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    prediction = prediction.clamp_min(0.0)
    pas_prediction = prediction[:, :1024].reshape(-1, 256, 4)
    pas_target = target[:, :1024].reshape(-1, 256, 4)
    pdp_prediction = prediction[:, 1024:].reshape(-1, 2, 4, 192)
    pdp_target = target[:, 1024:].reshape(-1, 2, 4, 192)
    pas_prediction /= torch.linalg.vector_norm(pas_prediction, dim=1, keepdim=True).clamp_min(1e-12)
    pas_target /= torch.linalg.vector_norm(pas_target, dim=1, keepdim=True).clamp_min(1e-12)
    pdp_prediction /= torch.linalg.vector_norm(pdp_prediction, dim=3, keepdim=True).clamp_min(1e-12)
    pdp_target /= torch.linalg.vector_norm(pdp_target, dim=3, keepdim=True).clamp_min(1e-12)
    pas = torch.sum(pas_prediction * pas_target, dim=1).mean().item()
    pdp = torch.sum(pdp_prediction * pdp_target, dim=3).mean().item()
    return pas, pdp


def main() -> None:
    args = parse_args()
    positions_all = np.load("Round1_Train_Pos.npy").astype(np.float32)
    contexts_all = np.load("artifacts/map_context.npz")["train"].astype(np.float32)
    features_all = np.asarray(
        np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32
    )
    valid = nonzero_feature_indices(features_all)
    inverse = np.full(len(features_all), -1, dtype=np.int64)
    inverse[valid] = np.arange(len(valid))
    positions, contexts = positions_all[valid], contexts_all[valid]
    features = torch.from_numpy(features_all[valid].copy()).cuda()
    embeddings = metric_embeddings(positions, contexts)
    aggregate: dict[str, list[tuple[float, float]]] = defaultdict(list)
    folds = []

    for suffix in FOLDS:
        checkpoint = torch.load(
            f"artifacts/cv_noout_split{suffix}.pt", map_location="cpu", weights_only=False
        )
        validation_global = np.asarray(checkpoint["validation_indices"], dtype=np.int64)[: args.limit]
        training_global = np.asarray(checkpoint["train_indices"], dtype=np.int64)
        query, training = inverse[validation_global], inverse[training_global]
        target = features[query]
        fold_scores = {}
        for metric in ("xy_y0.75", "xy_ctx-summary_s4", "xy_ctx-patch_s4"):
            local, distance = nearest_neighbors(
                embeddings[metric][query], embeddings[metric][training], args.max_neighbors
            )
            neighbor = training[local]
            source = features[neighbor]
            for neighbors in (8, 16, 32, 48):
                if neighbors > args.max_neighbors:
                    continue
                for power in (1.0, 2.0, 3.0):
                    for softening in (0.0, 1.0):
                        for ridge in (0.1, 1.0, 10.0):
                            weight = local_linear_weights(
                                positions[query, :2],
                                positions[neighbor[:, :neighbors], :2],
                                distance[:, :neighbors],
                                power,
                                softening,
                                ridge,
                            )
                            prediction = torch.einsum(
                                "qk,qkd->qd",
                                torch.from_numpy(weight).cuda(),
                                source[:, :neighbors],
                            )
                            pas, pdp = group_cosines(prediction, target)
                            name = (
                                f"{metric}_k{neighbors}_p{power:g}_e{softening:g}_r{ridge:g}"
                            )
                            fold_scores[name] = {"pas": pas, "pdp": pdp}
                            aggregate[name].append((pas, pdp))
        folds.append({"fold": suffix, "scores": fold_scores})
        print(f"completed fold {suffix}", flush=True)

    mean = {
        name: {
            "pas": float(np.mean([value[0] for value in values])),
            "pdp": float(np.mean([value[1] for value in values])),
        }
        for name, values in aggregate.items()
    }
    output = {
        "folds": folds,
        "top_pas": sorted(mean.items(), key=lambda item: item[1]["pas"], reverse=True)[:30],
        "top_pdp": sorted(mean.items(), key=lambda item: item[1]["pdp"], reverse=True)[:30],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output), encoding="utf-8")
    print(json.dumps({"top_pas": output["top_pas"][:10], "top_pdp": output["top_pdp"][:10]}, indent=2))


if __name__ == "__main__":
    main()
