from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.search_four_action_sample_gate import ACTIONS, FOLDS
from experiments.search_nonlinear_four_action_gate import load_data


def choose_biased_actions(
    value: np.ndarray,
    fraction: float,
    pdp_bias: float,
    both_bias: float,
) -> np.ndarray:
    adjusted = value + np.array([0.0, pdp_bias, both_bias])
    best_action = np.argmax(adjusted, axis=1).astype(np.int64) + 1
    benefit = np.max(adjusted, axis=1)
    count = int(round(fraction * len(value)))
    count = min(max(count, 0), len(value))
    selected = np.zeros(len(value), dtype=bool)
    if count:
        selected[np.argpartition(benefit, -count)[-count:]] = True
    return np.where(selected, best_action, 0)


def prepare_banks(stats: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        "pas": np.stack([stats[action]["final_pas"] for action in ACTIONS], axis=1),
        "pdp": np.stack([stats[action]["final_pdp"] for action in ACTIONS], axis=1),
        "cross": np.stack([stats[action]["final_cross"] for action in ACTIONS], axis=1),
        "pred_energy": np.stack(
            [stats[action]["final_pred_energy"] for action in ACTIONS], axis=1
        ),
        "target_energy": np.asarray(stats["none"]["target_energy"]),
    }


def metrics(bank: dict[str, np.ndarray], action: np.ndarray) -> dict[str, object]:
    row = np.arange(len(action))
    pas = float(bank["pas"][row, action].mean())
    pdp = float(bank["pdp"][row, action].mean())
    cross = bank["cross"][row, action].sum()
    pred_energy = bank["pred_energy"][row, action].sum()
    target_energy = bank["target_energy"].sum()
    nmse = float(
        1.0 - np.abs(cross) ** 2 / max(pred_energy * target_energy, 1e-30)
    )
    return {
        "pas": pas,
        "pdp": pdp,
        "nmse": nmse,
        "score": 0.4 * pas + 0.4 * pdp + 0.2 / (1.0 + nmse),
        "action_fraction": {
            name: float(np.mean(action == index))
            for index, name in enumerate(ACTIONS)
        },
    }


def evaluate(
    data: dict[str, dict[str, object]],
    values: dict[str, np.ndarray],
    banks: dict[str, dict[str, np.ndarray]],
    fraction: float,
    pdp_bias: float,
    both_bias: float,
) -> dict[str, object]:
    rows = []
    for fold in FOLDS:
        action = choose_biased_actions(values[fold], fraction, pdp_bias, both_bias)
        rows.append({"fold": fold, **metrics(banks[fold], action)})
    return {
        "score": float(np.mean([row["score"] for row in rows])),
        "pas": float(np.mean([row["pas"] for row in rows])),
        "pdp": float(np.mean([row["pdp"] for row in rows])),
        "nmse": float(np.mean([row["nmse"] for row in rows])),
        "folds": rows,
    }


def main() -> None:
    data = load_data()
    values = {
        fold: np.load(
            f"artifacts/v45_fine_xy_action_full_stats_split{fold}.npz"
        )["sample_spectral_action_value"].astype(np.float64)
        for fold in FOLDS
    }
    banks = {fold: prepare_banks(data[fold]["stats"]) for fold in FOLDS}

    coarse = np.linspace(-0.001, 0.001, 21)
    fractions = np.linspace(0.5, 1.0, 21)
    candidates = []
    for pdp_bias in coarse:
        for both_bias in coarse:
            for fraction in fractions:
                result = evaluate(
                    data, values, banks, fraction, pdp_bias, both_bias
                )
                candidates.append((result["score"], fraction, pdp_bias, both_bias, result))
    candidates.sort(reverse=True, key=lambda item: item[0])

    _, center_fraction, center_pdp, center_both, _ = candidates[0]
    fine_bias = np.linspace(-0.00012, 0.00012, 13)
    fine_fraction = np.linspace(-0.03, 0.03, 13)
    for pdp_delta in fine_bias:
        for both_delta in fine_bias:
            for fraction_delta in fine_fraction:
                fraction = float(np.clip(center_fraction + fraction_delta, 0.0, 1.0))
                pdp_bias = float(center_pdp + pdp_delta)
                both_bias = float(center_both + both_delta)
                result = evaluate(
                    data, values, banks, fraction, pdp_bias, both_bias
                )
                candidates.append((result["score"], fraction, pdp_bias, both_bias, result))
    candidates.sort(reverse=True, key=lambda item: item[0])

    def serialize(item: tuple) -> dict[str, object]:
        score, fraction, pdp_bias, both_bias, result = item
        return {
            "fraction": fraction,
            "bias": [0.0, pdp_bias, both_bias],
            **result,
        }

    output = {
        "best": serialize(candidates[0]),
        "top": [serialize(item) for item in candidates[:50]],
    }
    Path("artifacts/v46_action_bias_search.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"best": output["best"], "top": output["top"][:10]}, indent=2))


if __name__ == "__main__":
    main()
