"""Evaluate calibration and selective prediction for saved model outputs."""

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
    """Clip probabilities away from exactly 0 and 1."""
    return np.clip(
        np.asarray(probabilities, dtype=np.float64),
        EPSILON,
        1.0 - EPSILON,
    )


def probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    """Convert probabilities to logits."""
    probabilities = clip_probabilities(probabilities)
    return np.log(probabilities / (1.0 - probabilities))


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Apply a numerically stable sigmoid."""
    values = np.asarray(values, dtype=np.float64)

    output = np.empty_like(values)

    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))

    negative_values = values[~positive]
    exp_values = np.exp(negative_values)
    output[~positive] = exp_values / (1.0 + exp_values)

    return output


def apply_temperature(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Apply temperature scaling to binary probabilities."""
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    logits = probabilities_to_logits(probabilities)
    return sigmoid(logits / temperature)


def fit_temperature(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Fit temperature using validation log loss only."""
    labels = np.asarray(labels, dtype=np.int32)
    probabilities = clip_probabilities(probabilities)

    if labels.shape != probabilities.shape:
        raise ValueError(
            "Labels and probabilities must have matching shapes."
        )

    candidates = np.exp(
        np.linspace(
            np.log(0.05),
            np.log(20.0),
            2000,
        )
    )

    losses = np.asarray(
        [
            log_loss(
                labels,
                apply_temperature(
                    probabilities,
                    float(temperature),
                ),
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
    """Calculate expected calibration error using equal-width bins."""
    labels = np.asarray(labels, dtype=np.int32)
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if bins <= 0:
        raise ValueError("Number of bins must be positive.")

    if labels.shape != probabilities.shape:
        raise ValueError(
            "Labels and probabilities must have matching shapes."
        )

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    ece = 0.0

    for bin_index in range(bins):
        lower = edges[bin_index]
        upper = edges[bin_index + 1]

        if bin_index == bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(np.sum(mask))

        if count == 0:
            continue

        mean_confidence = float(
            np.mean(probabilities[mask])
        )

        observed_frequency = float(
            np.mean(labels[mask])
        )

        ece += (
            count / total
        ) * abs(
            mean_confidence - observed_frequency
        )

    return float(ece)


def calibrated_threshold(
    raw_threshold: float,
    temperature: float,
) -> float:
    """Transform an existing threshold through temperature scaling."""
    scaled = apply_temperature(
        np.asarray(
            [raw_threshold],
            dtype=np.float64,
        ),
        temperature,
    )

    return float(scaled[0])


def confidence_distance(
    probabilities: np.ndarray,
    decision_threshold: float,
) -> np.ndarray:
    """Return absolute distance from the decision boundary."""
    return np.abs(
        np.asarray(
            probabilities,
            dtype=np.float64,
        )
        - decision_threshold
    )


def validation_confidence_cutoff(
    probabilities: np.ndarray,
    decision_threshold: float,
    target_coverage: float,
) -> float:
    """Choose an abstention cutoff using validation predictions only."""

    confidence = confidence_distance(
        probabilities,
        decision_threshold,
    )

    quantile = 1.0 - target_coverage

    if target_coverage == 1.0:
        return 0.0
    else:
        return float(
        np.quantile(
            confidence,
            quantile,
        )
    )


def calibration_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 15,
) -> dict[str, float]:
    """Return calibration and ranking metrics."""
    probabilities = clip_probabilities(probabilities)

    return {
        "roc_auc": float(
            roc_auc_score(labels, probabilities)
        ),
        "brier_score": float(
            brier_score_loss(labels, probabilities)
        ),
        "log_loss": float(
            log_loss(
                labels,
                probabilities,
                labels=[0, 1],
            )
        ),
        "expected_calibration_error": (
            expected_calibration_error(
                labels,
                probabilities,
                bins=bins,
            )
        ),
    }


def selective_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    decision_threshold: float,
    confidence_cutoff: float,
) -> dict[str, float | int | None]:
    """Evaluate accepted predictions and audit deferred cases."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Labels and probabilities must be one-dimensional.")

    if labels.shape != probabilities.shape:
        raise ValueError("Labels and probabilities must have matching shapes.")

    if labels.size == 0:
        raise ValueError("At least one example is required.")

    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Labels must contain only 0 and 1.")

    if (
        not np.all(np.isfinite(probabilities))
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
    ):
        raise ValueError("Probabilities must be finite and within [0, 1].")

    if not np.isfinite(decision_threshold) or not 0.0 <= decision_threshold <= 1.0:
        raise ValueError("Decision threshold must be within [0, 1].")

    if not np.isfinite(confidence_cutoff) or confidence_cutoff < 0.0:
        raise ValueError("Confidence cutoff must be finite and non-negative.")

    labels = labels.astype(np.int32)

    confidence = confidence_distance(
        probabilities,
        decision_threshold,
    )

    accepted = confidence >= confidence_cutoff
    deferred = ~accepted

    # Calculate predictions for every image, including deferred images.
    predictions = (
        probabilities >= decision_threshold
    ).astype(np.int32)

    tumour = labels == 1
    non_tumour = labels == 0
    predicted_tumour = predictions == 1
    predicted_non_tumour = predictions == 0
    errors = predictions != labels

    total_count = int(labels.size)
    accepted_count = int(np.sum(accepted))
    deferred_count = int(np.sum(deferred))

    tumour_count = int(np.sum(tumour))
    non_tumour_count = int(np.sum(non_tumour))

    accepted_tumours = int(np.sum(accepted & tumour))
    accepted_non_tumours = int(np.sum(accepted & non_tumour))

    deferred_tumours = int(np.sum(deferred & tumour))
    deferred_non_tumours = int(np.sum(deferred & non_tumour))

    # Confusion counts for accepted cases only.
    true_positives = int(
        np.sum(accepted & tumour & predicted_tumour)
    )
    false_positives = int(
        np.sum(accepted & non_tumour & predicted_tumour)
    )
    true_negatives = int(
        np.sum(accepted & non_tumour & predicted_non_tumour)
    )
    false_negatives = int(
        np.sum(accepted & tumour & predicted_non_tumour)
    )

    accepted_errors = int(np.sum(accepted & errors))
    deferred_errors = int(np.sum(deferred & errors))
    total_errors = int(np.sum(errors))

    def safe_rate(numerator: int, denominator: int) -> float | None:
        """Return None when a rate has no valid denominator."""
        if denominator == 0:
            return None
        return float(numerator / denominator)

    return {
        "total_images": total_count,
        "accepted_images": accepted_count,
        "deferred_images": deferred_count,
        "coverage": float(accepted_count / total_count),

        # Existing accepted-case metrics.
        "accuracy": safe_rate(
            true_positives + true_negatives,
            accepted_count,
        ),
        "precision": safe_rate(
            true_positives,
            true_positives + false_positives,
        ),
        "recall": safe_rate(
            true_positives,
            accepted_tumours,
        ),

        # Accepted-case confusion counts.
        "accepted_true_positives": true_positives,
        "accepted_false_positives": false_positives,
        "accepted_true_negatives": true_negatives,
        "accepted_false_negatives": false_negatives,

        # Class counts and coverage.
        "total_tumour_images": tumour_count,
        "total_non_tumour_images": non_tumour_count,
        "accepted_tumour_images": accepted_tumours,
        "accepted_non_tumour_images": accepted_non_tumours,
        "deferred_tumour_images": deferred_tumours,
        "deferred_non_tumour_images": deferred_non_tumours,
        "tumour_coverage": safe_rate(
            accepted_tumours,
            tumour_count,
        ),
        "non_tumour_coverage": safe_rate(
            accepted_non_tumours,
            non_tumour_count,
        ),

        # Fraction of all tumour images automatically detected.
        "automatic_tumour_detection_rate": safe_rate(
            true_positives,
            tumour_count,
        ),

        # Audit whether deferral captures difficult cases.
        "accepted_errors": accepted_errors,
        "deferred_errors_before_deferral": deferred_errors,
        "accepted_error_rate": safe_rate(
            accepted_errors,
            accepted_count,
        ),
        "deferred_error_rate_before_deferral": safe_rate(
            deferred_errors,
            deferred_count,
        ),
        "fraction_of_all_errors_deferred": safe_rate(
            deferred_errors,
            total_errors,
        ),
    }


def build_selective_results(
    validation_probabilities: np.ndarray,
    test_labels: np.ndarray,
    test_probabilities: np.ndarray,
    decision_threshold: float,
    coverages: list[float],
) -> list[dict]:
    """Evaluate validation-derived abstention cutoffs on the test set."""
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


def calibration_bins(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Return average prediction and observed rate for non-empty bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)

    predicted = []
    observed = []

    for bin_index in range(bins):
        lower = edges[bin_index]
        upper = edges[bin_index + 1]

        if bin_index == bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        if not np.any(mask):
            continue

        predicted.append(
            float(np.mean(probabilities[mask]))
        )
        observed.append(
            float(np.mean(labels[mask]))
        )

    return (
        np.asarray(predicted),
        np.asarray(observed),
    )


def save_reliability_curve(
    labels: np.ndarray,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    output_path: Path,
    bins: int = 15,
) -> None:
    """Save raw and calibrated reliability curves."""
    raw_predicted, raw_observed = calibration_bins(
        labels,
        raw_probabilities,
        bins=bins,
    )

    calibrated_predicted, calibrated_observed = calibration_bins(
        labels,
        calibrated_probabilities,
        bins=bins,
    )

    figure, axis = plt.subplots(figsize=(6.5, 5.5))

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    axis.plot(
        raw_predicted,
        raw_observed,
        marker="o",
        label="Raw",
    )

    axis.plot(
        calibrated_predicted,
        calibrated_observed,
        marker="o",
        label="Temperature scaled",
    )

    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed positive frequency")
    axis.set_title("Reliability curve")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.2)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_selective_prediction_plot(
    results: list[dict],
    output_path: Path,
) -> None:
    """Save coverage-versus-accuracy plot."""
    coverage = [
        100.0 * result["coverage"]
        for result in results
    ]

    accuracy = [
        100.0 * result["accuracy"]
        for result in results
    ]

    figure, axis = plt.subplots(figsize=(6.5, 5.0))

    axis.plot(
        coverage,
        accuracy,
        marker="o",
        linewidth=2,
    )

    axis.set_xlabel("Coverage (%)")
    axis.set_ylabel("Accuracy on retained cases (%)")
    axis.set_title("Selective prediction")
    axis.grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def parse_coverages(raw: str) -> list[float]:
    """Parse comma-separated coverage values."""
    coverages = [
        float(value.strip())
        for value in raw.split(",")
        if value.strip()
    ]

    if not coverages:
        raise ValueError(
            "At least one coverage value is required."
        )

    if any(
        not 0.0 < coverage <= 1.0
        for coverage in coverages
    ):
        raise ValueError(
            "Coverage values must be in (0, 1]."
        )

    return coverages


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

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
        default=(
            ROOT
            / "artifacts"
            / "calibration"
        ),
    )

    parser.add_argument(
        "--figure-directory",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "figures"
        ),
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--coverages",
        default="1.0,0.95,0.90,0.80",
    )

    return parser.parse_args()


def main() -> None:
    """Run calibration and selective-prediction evaluation."""
    args = parse_args()

    if not args.probabilities.is_file():
        raise FileNotFoundError(
            f"Missing probability file: {args.probabilities}"
        )

    if not args.evaluation.is_file():
        raise FileNotFoundError(
            f"Missing evaluation file: {args.evaluation}"
        )

    saved = np.load(
        args.probabilities,
        allow_pickle=False,
    )

    required = {
        "y_val",
        "validation_probabilities",
        "y_test",
        "test_probabilities",
    }

    missing = required.difference(saved.files)

    if missing:
        raise ValueError(
            "Probability file is missing arrays: "
            + ", ".join(sorted(missing))
        )

    validation_labels = (
        saved["y_val"].astype(np.int32)
    )

    validation_probabilities = (
        saved["validation_probabilities"]
        .astype(np.float64)
    )

    test_labels = (
        saved["y_test"].astype(np.int32)
    )

    test_probabilities = (
        saved["test_probabilities"]
        .astype(np.float64)
    )

    with args.evaluation.open(
        encoding="utf-8"
    ) as input_file:
        evaluation = json.load(input_file)

    raw_threshold = float(
        evaluation["selected_threshold"]
    )

    temperature = fit_temperature(
        validation_labels,
        validation_probabilities,
    )

    calibrated_validation = apply_temperature(
        validation_probabilities,
        temperature,
    )

    calibrated_test = apply_temperature(
        test_probabilities,
        temperature,
    )

    new_threshold = calibrated_threshold(
        raw_threshold,
        temperature,
    )

    coverages = parse_coverages(
        args.coverages
    )

    selective_results = build_selective_results(
        calibrated_validation,
        test_labels,
        calibrated_test,
        new_threshold,
        coverages,
    )

    report = {
        "calibration_method": "temperature_scaling",
        "fit_split": "validation",
        "evaluation_split": "test",
        "temperature": temperature,
        "raw_threshold": raw_threshold,
        "calibrated_threshold": new_threshold,
        "validation": {
            "raw": calibration_metrics(
                validation_labels,
                validation_probabilities,
                bins=args.bins,
            ),
            "calibrated": calibration_metrics(
                validation_labels,
                calibrated_validation,
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
                calibrated_test,
                bins=args.bins,
            ),
        },
        "selective_prediction": selective_results,
    }

    args.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.figure_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        args.output_directory
        / "calibration_report.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            report,
            output_file,
            indent=2,
        )

    save_reliability_curve(
        test_labels,
        test_probabilities,
        calibrated_test,
        (
            args.figure_directory
            / "reliability_curve.png"
        ),
        bins=args.bins,
    )

    save_selective_prediction_plot(
        selective_results,
        (
            args.figure_directory
            / "selective_prediction.png"
        ),
    )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    print(
        f"Saved report to {report_path}"
    )


if __name__ == "__main__":
    main()
