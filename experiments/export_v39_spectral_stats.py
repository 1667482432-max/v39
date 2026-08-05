from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.evaluate_v5_phase import load_fold_expert_gate
from physical_ai.data import RoundData
from physical_ai.features import nonzero_feature_indices
from predict import GROUP_CONFIGS, load_model, predict_spectral_features


FOLDS = ("101", "202", "20260804", "303", "404")


def cosine_parts(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    pas_prediction = prediction[:, :1024].reshape(-1, 256, 4)
    pas_target = target[:, :1024].reshape(-1, 256, 4)
    pdp_prediction = prediction[:, 1024:].reshape(-1, 2, 4, 192)
    pdp_target = target[:, 1024:].reshape(-1, 2, 4, 192)
    pas = torch.nn.functional.cosine_similarity(
        pas_prediction, pas_target, dim=1
    ).mean()
    pdp = torch.nn.functional.cosine_similarity(
        pdp_prediction, pdp_target, dim=-1
    ).mean()
    return float(pas.item()), float(pdp.item())


@torch.inference_mode()
def main() -> None:
    device = torch.device("cuda")
    data = RoundData(Path("."))
    positions_all = np.asarray(data.train_positions, dtype=np.float32)
    contexts_all = np.load("artifacts/map_context_advanced.npz")["train"].astype(
        np.float32
    )
    features_all = np.asarray(
        np.load("artifacts/spectral_features.npy", mmap_mode="r"), dtype=np.float32
    )
    valid = nonzero_feature_indices(features_all)
    inverse = np.full(len(features_all), -1, dtype=np.int64)
    inverse[valid] = np.arange(len(valid))
    positions = positions_all[valid]
    contexts = contexts_all[valid]
    features = torch.from_numpy(features_all[valid].copy()).to(device)
    groupwise = json.loads(
        Path("artifacts/groupwise_kriging_v2.json").read_text(encoding="utf-8")
    )
    conditional = json.loads(
        Path("artifacts/conditional_groupwise_lowreg.json").read_text(
            encoding="utf-8"
        )
    )
    regularization_key = f'{float(conditional["final"]["regularization"]):g}'
    rows = []
    for fold in FOLDS:
        checkpoint_path = Path(f"artifacts/cv_noout_split{fold}.pt")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        val_global = np.asarray(checkpoint["validation_indices"], dtype=np.int64)
        train_global = np.asarray(checkpoint["train_indices"], dtype=np.int64)
        val_idx = inverse[val_global]
        train_idx = inverse[train_global]
        fold_groupwise = next(
            row for row in groupwise["loo"] if row["fold"] == fold
        )
        group_weights = torch.tensor(
            fold_groupwise["weights"], dtype=torch.float32, device=device
        )
        fold_conditional = next(
            row
            for row in conditional["regularizations"][regularization_key]
            if row["fold"] == fold
        )
        conditional_meta = {"final": fold_conditional}
        pas_gate = load_fold_expert_gate(
            Path("artifacts/disagreement_gating_cv_pas_advanced_s075_e3.pt"),
            "pas",
            fold,
            device,
        )
        pdp_gate = load_fold_expert_gate(
            Path("artifacts/disagreement_gating_cv_pdp_advanced_s06_e2.pt"),
            "pdp",
            fold,
            device,
        )
        model = load_model(checkpoint_path, device)
        prediction = predict_spectral_features(
            positions[train_idx],
            positions[val_idx],
            contexts[train_idx],
            contexts[val_idx],
            features[train_idx],
            [model],
            0.0001,
            8,
            group_weights,
            conditional_meta,
            GROUP_CONFIGS,
            pas_gate,
            pdp_gate,
        )
        target = features[val_idx]
        pas, pdp = cosine_parts(prediction, target)
        output = Path(f"artifacts/v39_spectral_stats_split{fold}.npz")
        np.savez_compressed(
            output,
            global_index=val_global,
            position=positions[val_idx],
            context=contexts[val_idx],
            prediction=prediction.cpu().numpy().astype(np.float32),
            target=target.cpu().numpy().astype(np.float32),
        )
        row = {"fold": fold, "pas": pas, "pdp": pdp, "output": str(output)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = {
        "folds": rows,
        "pas": float(np.mean([row["pas"] for row in rows])),
        "pdp": float(np.mean([row["pdp"] for row in rows])),
    }
    Path("artifacts/v39_spectral_stats.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
