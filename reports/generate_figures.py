"""Generate portfolio figures from saved evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve

ROOT = Path(__file__).resolve().parents[1]


def read_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as config_file:
        return json.load(config_file)


def save_confusion_matrix(config: dict, output_directory: Path) -> None:
    metrics = config["tuned_test_metrics"]
    matrix = np.array(
        [
            [metrics["true_negatives"], metrics["false_positives"]],
            [metrics["false_negatives"], metrics["true_positives"]],
        ]
    )

    figure, axis = plt.subplots(figsize=(6.6, 5.6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    for row in range(2):
        for column in range(2):
            color = "white" if matrix[row, column] > matrix.max() / 2 else "#172033"
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color=color,
            )

    axis.set_xticks([0, 1], labels=["No tumour", "Tumour"])
    axis.set_yticks([0, 1], labels=["No tumour", "Tumour"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Tuned-threshold test confusion matrix")
    figure.tight_layout()
    figure.savefig(output_directory / "confusion_matrix_tuned.png", dpi=200)
    plt.close(figure)


def save_subtype_chart(config: dict, output_directory: Path) -> None:
    subtype_results = config["tuned_subtype_results"]
    display_names = {
        "glioma": "Glioma",
        "meningioma": "Meningioma",
        "pituitary": "Pituitary",
        "notumor": "No tumour",
    }
    names = [display_names[result["category"]] for result in subtype_results]
    accuracies = [100 * result["accuracy"] for result in subtype_results]

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    colors = ["#2F6BFF", "#5A83F1", "#6CAED6", "#63A46C"]
    bars = axis.bar(names, accuracies, color=colors, width=0.68)
    axis.set_ylim(0, 105)
    axis.set_ylabel("Correctly classified (%)")
    axis.set_title("Test performance by source category")
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)

    for bar, accuracy in zip(bars, accuracies, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            accuracy + 0.45,
            f"{accuracy:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    figure.tight_layout()
    figure.savefig(output_directory / "subtype_performance.png", dpi=200)
    plt.close(figure)


def save_roc_curve(probability_path: Path, output_directory: Path) -> None:
    saved = np.load(probability_path, allow_pickle=False)
    labels = saved["y_test"]
    probabilities = saved["test_probabilities"]

    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    score = auc(false_positive_rate, true_positive_rate)

    figure, axis = plt.subplots(figsize=(6.6, 5.6))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        color="#2457D6",
        linewidth=2.5,
        label=f"VGG16 classifier (AUC = {score:.4f})",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#7A8499", label="Random")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.01)
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title("Held-out test ROC curve")
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_directory / "roc_curve.png", dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "artifacts" / "threshold_evaluation.json",
    )
    parser.add_argument(
        "--probabilities",
        type=Path,
        default=ROOT / "artifacts" / "evaluation_probabilities.npz",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "docs" / "figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    config = read_config(args.config)

    save_confusion_matrix(config, args.output_directory)
    save_subtype_chart(config, args.output_directory)

    if args.probabilities.is_file():
        save_roc_curve(args.probabilities, args.output_directory)
        print("Generated confusion matrix, subtype chart, and ROC curve.")
    else:
        print(
            "Generated confusion matrix and subtype chart. "
            f"ROC curve skipped because {args.probabilities} is missing."
        )


if __name__ == "__main__":
    main()
