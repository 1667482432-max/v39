from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from physical_ai.data import RoundData
from physical_ai.features import nonzero_feature_indices
from physical_ai.neighbors import nearest_neighbors
from physical_ai.spatial import ADVANCED_MAP_METRIC, metric_embeddings


FOLDS = ("101", "202", "20260804", "303", "404")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive exact post-group-energy per-UE validation statistics"
    )
    parser.add_argument(
        "--stats-pattern", default="artifacts/v37_scalar_stats_split{fold}.npz"
    )
    parser.add_argument(
        "--output-pattern", default="artifacts/v37_ue_stats_split{fold}.npz"
    )
    parser.add_argument(
        "--checkpoint-pattern", default="artifacts/cv_noout_split{fold}.pt"
    )
    parser.add_argument(
        "--context", type=Path, default=Path("artifacts/map_context_advanced.npz")
    )
    parser.add_argument(
        "--group-energy", type=Path, default=Path("artifacts/channel_group_energy.npy")
    )
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--power", type=float, default=4.0)
    parser.add_argument("--legacy-metric", default="xy_ctx-patch_s4")
    parser.add_argument("--advanced-metric", default=ADVANCED_MAP_METRIC)
    parser.add_argument("--advanced-min-distance", type=float, default=1.2)
    parser.add_argument("--advanced-max-distance", type=float, default=4.3)
    parser.add_argument("--strength", type=float, default=0.3)
    parser.add_argument("--mid-strength", type=float, default=0.5)
    parser.add_argument("--mid-min-distance", type=float, default=1.6)
    parser.add_argument("--mid-max-distance", type=float, default=3.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = RoundData(Path("."))
    positions_all = np.asarray(data.train_positions, dtype=np.float32)
    contexts_all = np.load(args.context)["train"].astype(np.float32)
    features = np.load("artifacts/spectral_features.npy", mmap_mode="r")
    valid_global = nonzero_feature_indices(features)
    inverse = np.full(len(features), -1, dtype=np.int64)
    inverse[valid_global] = np.arange(len(valid_global))
    positions = positions_all[valid_global]
    contexts = contexts_all[valid_global]
    embeddings = metric_embeddings(positions, contexts)
    all_group_energy = np.load(args.group_energy)
    group_fraction = all_group_energy / np.maximum(
        all_group_energy.sum(axis=(1, 2), keepdims=True), 1e-30
    )

    for fold in FOLDS:
        stats_path = Path(args.stats_pattern.format(fold=fold))
        stats = dict(np.load(stats_path))
        checkpoint = torch.load(
            args.checkpoint_pattern.format(fold=fold),
            map_location="cpu",
            weights_only=False,
        )
        val_global = np.asarray(checkpoint["validation_indices"], dtype=np.int64)[
            : len(stats["global_index"])
        ]
        train_global = np.asarray(checkpoint["train_indices"], dtype=np.int64)
        if not np.array_equal(val_global, stats["global_index"]):
            raise ValueError(f"Validation order mismatch for fold {fold}")
        val_idx = inverse[val_global]
        train_idx = inverse[train_global]
        _, physical_distance = nearest_neighbors(
            positions[val_idx], positions[train_idx], 16
        )

        def interpolate(metric: str) -> np.ndarray:
            coordinates = embeddings[metric]
            local, distance = nearest_neighbors(
                coordinates[val_idx], coordinates[train_idx], args.neighbors
            )
            weight = (distance + 1e-3) ** (-args.power)
            weight /= weight.sum(axis=1, keepdims=True)
            prediction = np.exp(
                np.sum(
                    weight[:, :, None, None]
                    * np.log(group_fraction[train_global[local]].clip(1e-12)),
                    axis=1,
                )
            )
            return prediction / prediction.sum(axis=(1, 2), keepdims=True)

        legacy = interpolate(args.legacy_metric)
        advanced = interpolate(args.advanced_metric)
        nearest = physical_distance[:, 0]
        use_advanced = (nearest >= args.advanced_min_distance) & (
            nearest < args.advanced_max_distance
        )
        desired = np.where(use_advanced[:, None, None], advanced, legacy)
        strength = np.where(
            (nearest >= args.mid_min_distance) & (nearest < args.mid_max_distance),
            args.mid_strength,
            args.strength,
        )
        current_energy = stats["pred_energy_pol_ue"].astype(np.float64)
        current_fraction = current_energy / np.maximum(
            current_energy.sum(axis=(1, 2), keepdims=True), 1e-30
        )
        amplitude_scale = (desired / np.maximum(current_fraction, 1e-12)) ** (
            strength[:, None, None] / 2.0
        )
        final_cross_pol_ue = amplitude_scale * stats["cross_pol_ue"]
        final_pred_energy_pol_ue = amplitude_scale**2 * current_energy
        final_cross_ue = final_cross_pol_ue.sum(axis=1)
        final_pred_energy_ue = final_pred_energy_pol_ue.sum(axis=1)
        cross_error = np.max(
            np.abs(final_cross_ue.sum(axis=1) - stats["final_cross"])
            / np.maximum(np.abs(stats["final_cross"]), 1e-30)
        )
        energy_error = np.max(
            np.abs(final_pred_energy_ue.sum(axis=1) - stats["final_pred_energy"])
            / np.maximum(stats["final_pred_energy"], 1e-30)
        )
        if cross_error > 2e-5 or energy_error > 2e-5:
            raise ValueError(
                f"Post-group derivation mismatch for {fold}: "
                f"cross={cross_error:g}, energy={energy_error:g}"
            )
        stats.update(
            {
                "final_cross_ue": final_cross_ue,
                "final_pred_energy_ue": final_pred_energy_ue,
                "final_cross_pol_ue": final_cross_pol_ue,
                "final_pred_energy_pol_ue": final_pred_energy_pol_ue,
                "predicted_group_fraction": desired,
                "group_energy_amplitude_scale": amplitude_scale,
            }
        )
        output = Path(args.output_pattern.format(fold=fold))
        np.savez_compressed(output, **stats)
        print(
            f"{fold}: {output} cross_error={cross_error:.3g} "
            f"energy_error={energy_error:.3g}"
        )


if __name__ == "__main__":
    main()
