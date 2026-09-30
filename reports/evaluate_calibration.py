"""Evaluate probability calibration and selective prediction.

Temperature scaling is fit using validation predictions only. The fitted
temperature is then locked and applied to the held-out test predictions.

The script also evaluates selective prediction by deriving uncertainty
cutoffs from validation predictions and applying those cutoffs unchanged
to the test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
EPSILON = 1e-7


def clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Keep probabilities away from exactly 0 and 1 for stable logits."""
    return np.clip(
        np.asarray(probabilities, dtype=np.float64),
        EPSILON,
        1.0 - EPSILON,
    )


def probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    """Convert binary probabilities to logits."""
    probabilities = clip_probabilities(probabilities)
    return np.log(probabilities / (1.0 - probabilities))


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)

    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))

    negative_values = values[~positive]
    exp_values = np.exp(negative_values)
    output[~positive] = exp_values / (1.0 + exp_values)

    return output


def apply_temperature(probabilities: np.ndarray, temperature: float,) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    logits = probabilities_to_logits(probabilities)
    return sigmoid(logits / temperature)


def fit_temperature(labels: np.ndarray, probabilities: np.ndarray,) -> float:
    labels = np.asarray(labels, dtype=np.int32)
    probabilities = clip_probabilities(probabilities)

    if labels.shape != probabilities.shape:
        raise ValueError("Labels and probabilities must have matching shapes.")

    # Search temperatures from approximately 0.05 to 20.
    candidates = np.exp(np.linspace(np.log(0.05), np.log(20.0), 2000))

    losses = np.asarray(
        [
            log_loss(
                labels,
                apply_temperature(probabilities, float(temperature)),
                labels=[0, 1],
            )
            for temperature in candidates
        ]
    )

    best_index = int(np.argmin(losses))
    return float(candidates[best_index])


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 15,
) -> float:
    """Calculate equal-width expected calibration error."""
    labels = np.asarray(labels, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if bins <= 0:
        raise ValueError("Number of bins must be positive.")

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    total = len(labels)

    for bin_index in range(bins):
        lower = edges[bin_index]
        upper = edges[bin_index + 1]

        if bin_index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)

        count = int(np.sum(mask))
        if count == 0:
            continue

        mean_confidence = float(np.mean(probabilities[mask]))
        observed_frequency = float(np.mean(labels[mask]))

        ece += (count / total) * abs(mean_confidence - observed_frequency)

    return float(ece)


def calibration_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 15,
) -> dict[str, float]:
    """Return score-level calibration metrics."""
    probabilities = clip_probabilities(probabilities)

    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "expected_calibration_error": expected_calibration_error(
            labels,
            probabilities,
            bins=bins,
        ),
    }


def calibration_bins(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bin confidence, observed frequency, and counts."""
    edges = np.linspace(0.0, 1.0, bins + 1)

    mean_probabilities = []
    positive_frequencies = []
    counts = []

    for bin_index in range(bins):
        lower = edges[bin_index]
        upper = edges[bin_index + 1]

        if bin_index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)

        if not np.any(mask):
            continue

        mean_probabilities.append(float(np.mean(probabilities[mask])))
        positive_frequencies.append(float(np.mean(labels[mask])))
        counts.append(int(np.sum(mask)))

    return (
        np.asarray(mean_probabilities),
        np.asarray(positive_frequencies),
        np.asarray(counts),
    )


def save_reliability_curve(
    labels: np.ndarray,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    output_path: Path,
    bins: int,
) -> None:
    """Plot raw and calibrated reliability curves."""
    raw_confidence, raw_frequency, _ = calibration_bins(
        labels,
        raw_probabilities,
        bins,
    )
    calibrated_confidence, calibrated_frequency, _ = calibration_bins(
        labels,
        calibrated_probabilities,
        bins,
    )

    figure, axis = plt.subplots(figsize=(6.8, 5.8))

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Perfect calibration",
    )
    axis.plot(
        raw_confidence,
        raw_frequency,
        marker="o",
        linewidth=2,
        label="Raw scores",
    )
    axis.plot(
        calibrated_confidence,
        calibrated_frequency,
        marker="o",
        linewidth=2,
        label="Temperature-scaled",
    )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed tumour frequency")
    axis.set_title("Held-out test reliability curve")
    axis.grid(alpha=0.2)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def calibrated_threshold(
    raw_threshold: float,
    temperature: float,
) -> float:
    """Transform the original decision threshold after calibration.

    Because temperature scaling is monotonic, transforming the threshold
    preserves the model's original binary decisions.
    """
    scaled = apply_temperature(
        np.asarray([raw_threshold], dtype=np.float64),
        temperature,
    )
    return float(scaled[0])


def confidence_distance(
    probabilities: np.ndarray,
    decision_threshold: float,
) -> np.ndarray:
    """Measure confidence as distance from the locked decision boundary."""
    return np.abs(
        np.asarray(probabilities, dtype=np.float64) - decision_threshold
    )


def validation_confidence_cutoff(
    probabilities: np.ndarray,
    decision_threshold: float,
    target_coverage: float,
) -> float:
    """Choose an abstention cutoff using validation predictions only."""
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("Coverage must be in the interval (0, 1].")

    confidence = confidence_distance(probabilities, decision_threshold)

    # Retain approximately the requested proportion of most-confident cases.
    quantile = 1.0 - target_coverage
    return float(np.quantile(confidence, quantile))


def selective_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    decision_threshold: float,
    confidence_cutoff: float,
) -> dict[str, float | int]:
    """Evaluate test cases retained after validation-derived abstention."""
    confidence = confidence_distance(probabilities, decision_threshold)
    accepted = confidence >= confidence_cutoff

    accepted_count = int(np.sum(accepted))
    total_count = int(len(labels))

    if accepted_count == 0:
        raise ValueError("Selective prediction rejected every test example.")

    accepted_labels = labels[accepted]
    accepted_probabilities = probabilities[accepted]
    accepted_predictions = (
        accepted_probabilities >= decision_threshold
    ).astype(np.int32)

    return {
        "accepted_images": accepted_count,
        "deferred_images": total_count - accepted_count,
        "coverage": float(accepted_count / total_count),
        "accuracy": float(
            accuracy_score(accepted_labels, accepted_predictions)
        ),
        "precision": float(
            precision_score(
                accepted_labels,
                accepted_predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                accepted_labels,
                accepted_predictions,
                zero_division=0,
            )
        ),
    }


def build_selective_results(
    validation_probabilities: np.ndarray,
    test_labels: np.ndarray,
    test_probabilities: np.ndarray,
    decision_threshold: float,
    coverages: list[float],
) -> list[dict]:
    """Apply validation-derived confidence cutoffs to test predictions."""
    results = []

    for target_coverage in coverages:
        cutoff = validation_confidence_cutoff(
            validation_probabilities,
            decision_threshold,
            target_coverage,
        )

        metrics = selective_metrics(
            test_labels,
            test_probabilities,
            decision_threshold,
            cutoff,
        )

        results.append(
            {
                "target_validation_coverage": target_coverage,
                "validation_confidence_cutoff": cutoff,
                **metrics,
            }
        )

    return results


def save_selective_prediction_plot(
    selective_results: list[dict],
    output_path: Path,
) -> None:
    """Plot test accuracy against retained coverage."""
    coverage = [
        100.0 * result["coverage"]
        for result in selective_results
    ]
    accuracy = [
        100.0 * result["accuracy"]
        for result in selective_results
    ]

    figure, axis = plt.subplots(figsize=(7.2, 5.2))

    axis.plot(
        coverage,
        accuracy,
        marker="o",
        linewidth=2.2,
    )

    for x_value, y_value in zip(coverage, accuracy, strict=True):
        axis.annotate(
            f"{y_value:.1f}%",
            (x_value, y_value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
        )

    axis.set_xlabel("Test coverage (%)")
    axis.set_ylabel("Accuracy on retained cases (%)")
    axis.set_title("Selective prediction: coverage vs. accuracy")
    axis.grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def parse_coverages(raw_coverages: str) -> list[float]:
    """Parse comma-separated coverage values."""
    coverages = [
        float(value.strip())
        for value in raw_coverages.split(",")
        if value.strip()
    ]

    if not coverages:
        raise ValueError("At least one coverage value is required.")

    if any(not 0.0 < coverage <= 1.0 for coverage in coverages):
        raise ValueError("Every coverage value must be in (0, 1].")

    return coverages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit temperature scaling on validation predictions and "
            "evaluate calibration/selective prediction on held-out test data."
        )
    )

    parser.add_argument(
        "--probabilities",
        type=Path,
        default=(
            ROOT
            / "artifacts"
            / "clean_finetuned"
            / "probabilities.npz"
        ),
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=(
            ROOT
            / "artifacts"
            / "clean_finetuned"
            / "evaluation.json"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "artifacts" / "calibration",
    )
    parser.add_argument(
        "--figure-directory",
        type=Path,
        default=ROOT / "docs" / "figures",
    )
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument(
        "--coverages",
        default="1.0,0.95,0.90,0.80",
        help="Comma-separated retained coverage targets.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.probabilities.is_file():
        raise FileNotFoundError(
            f"Prediction artifact does not exist: {args.probabilities}"
        )

    if not args.evaluation.is_file():
        raise FileNotFoundError(
            f"Evaluation report does not exist: {args.evaluation}"
        )

    saved = np.load(args.probabilities, allow_pickle=False)

    required_arrays = {
        "y_val",
        "validation_probabilities",
        "y_test",
        "test_probabilities",
    }
    missing_arrays = required_arrays.difference(saved.files)

    if missing_arrays:
        raise ValueError(
            "Probability artifact is missing arrays: "
            + ", ".join(sorted(missing_arrays))
        )

    validation_labels = saved["y_val"].astype(np.int32)
    validation_probabilities = saved[
        "validation_probabilities"
    ].astype(np.float64)

    test_labels = saved["y_test"].astype(np.int32)
    test_probabilities = saved[
        "test_probabilities"
    ].astype(np.float64)

    with args.evaluation.open(encoding="utf-8") as input_file:
        evaluation = json.load(input_file)

    raw_threshold = float(evaluation["selected_threshold"])

    temperature = fit_temperature(
        validation_labels,
        validation_probabilities,
    )

    calibrated_validation_probabilities = apply_temperature(
        validation_probabilities,
        temperature,
    )
    calibrated_test_probabilities = apply_temperature(
        test_probabilities,
        temperature,
    )

    transformed_threshold = calibrated_threshold(
        raw_threshold,
        temperature,
    )

    coverages = parse_coverages(args.coverages)

    selective_results = build_selective_results(
        calibrated_validation_probabilities,
        test_labels,
        calibrated_test_probabilities,
        transformed_threshold,
        coverages,
    )

    report = {
        "source_probabilities": str(args.probabilities),
        "source_evaluation": str(args.evaluation),
        "calibration_method": "temperature_scaling",
        "calibration_fit_split": "validation",
        "evaluation_split": "test",
        "temperature": temperature,
        "raw_decision_threshold": raw_threshold,
        "calibrated_decision_threshold": transformed_threshold,
        "validation": {
            "raw": calibration_metrics(
                validation_labels,
                validation_probabilities,
                bins=args.bins,
            ),
            "calibrated": calibration_metrics(
                validation_labels,
                calibrated_validation_probabilities,
                bins=args.bins,
            ),
        },
        "test": {
            "raw": calibration_metrics(
                test_labels,
                test_probabilities,
                bins=args.bins,
            ),
            "calibrated": calibration_metrics(
                test_labels,
                calibrated_test_probabilities,
                bins=args.bins,
            ),
        },
        "selective_prediction": selective_results,
    }

    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.figure_directory.mkdir(parents=True, exist_ok=True)

    report_path = args.output_directory / "calibration_report.json"

    with report_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)

    np.savez_compressed(
        args.output_directory / "calibrated_probabilities.npz",
        y_val=validation_labels,
        raw_validation_probabilities=validation_probabilities,
        calibrated_validation_probabilities=(
            calibrated_validation_probabilities
        ),
        y_test=test_labels,
        raw_test_probabilities=test_probabilities,
        calibrated_test_probabilities=calibrated_test_probabilities,
    )

    save_reliability_curve(
        test_labels,
        test_probabilities,
        calibrated_test_probabilities,
        args.figure_directory / "reliability_curve.png",
        args.bins,
    )

    save_selective_prediction_plot(
        selective_results,
        args.figure_directory / "selective_prediction.png",
    )

    print(json.dumps(report, indent=2))
    print(f"Saved report to {report_path}")
    print(
        "Saved figures to "
        f"{args.figure_directory / 'reliability_curve.png'} and "
        f"{args.figure_directory / 'selective_prediction.png'}"
    )


if __name__ == "__main__":
    main()
