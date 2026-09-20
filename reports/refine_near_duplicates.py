"""Refine perceptual-hash candidates with full-image structural similarity."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"glioma": 1, "meningioma": 1, "pituitary": 1, "notumor": 0}
FIELDNAMES = [
    "hamming_distance",
    "structural_similarity",
    "normalized_rmse",
    "left_split",
    "left_category",
    "left_path",
    "right_split",
    "right_category",
    "right_path",
    "same_category",
    "same_binary_label",
]


@lru_cache(maxsize=512)
def normalized_grayscale(path_string: str) -> np.ndarray:
    """Load one image as a fixed-size grayscale float array."""
    image = cv2.imread(path_string, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path_string}")
    return cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA).astype(
        np.float64
    )


def structural_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Calculate mean SSIM using the standard Gaussian-window formulation."""
    constant_1 = (0.01 * 255.0) ** 2
    constant_2 = (0.03 * 255.0) ** 2

    mean_left = cv2.GaussianBlur(left, (11, 11), 1.5)
    mean_right = cv2.GaussianBlur(right, (11, 11), 1.5)
    mean_left_squared = mean_left * mean_left
    mean_right_squared = mean_right * mean_right
    mean_product = mean_left * mean_right

    variance_left = cv2.GaussianBlur(left * left, (11, 11), 1.5) - mean_left_squared
    variance_right = (
        cv2.GaussianBlur(right * right, (11, 11), 1.5) - mean_right_squared
    )
    covariance = cv2.GaussianBlur(left * right, (11, 11), 1.5) - mean_product

    numerator = (2 * mean_product + constant_1) * (2 * covariance + constant_2)
    denominator = (
        mean_left_squared + mean_right_squared + constant_1
    ) * (variance_left + variance_right + constant_2)
    similarity_map = numerator / np.maximum(denominator, np.finfo(float).eps)
    return float(np.mean(similarity_map))


def normalized_rmse(left: np.ndarray, right: np.ndarray) -> float:
    """Return root-mean-square pixel error on a zero-to-one scale."""
    return float(np.sqrt(np.mean((left - right) ** 2)) / 255.0)


def refine_row(row: dict[str, str]) -> dict[str, str | int | float]:
    left = normalized_grayscale(row["left_path"])
    right = normalized_grayscale(row["right_path"])
    left_label = LABELS[row["left_category"]]
    right_label = LABELS[row["right_category"]]

    return {
        "hamming_distance": int(row["hamming_distance"]),
        "structural_similarity": structural_similarity(left, right),
        "normalized_rmse": normalized_rmse(left, right),
        "left_split": row["left_split"],
        "left_category": row["left_category"],
        "left_path": row["left_path"],
        "right_split": row["right_split"],
        "right_category": row["right_category"],
        "right_path": row["right_path"],
        "same_category": int(row["left_category"] == row["right_category"]),
        "same_binary_label": int(left_label == right_label),
    }


def threshold_summary(rows: list[dict], threshold: float) -> dict:
    selected = [row for row in rows if row["structural_similarity"] >= threshold]
    return {
        "pair_count": len(selected),
        "split_pairs": {
            "|".join(pair): count
            for pair, count in Counter(
                (row["left_split"], row["right_split"]) for row in selected
            ).items()
        },
        "different_category_pairs": sum(
            not row["same_category"] for row in selected
        ),
        "conflicting_binary_label_pairs": sum(
            not row["same_binary_label"] for row in selected
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine dHash candidates using structural similarity."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "artifacts" / "leakage_audit" / "near_duplicate_pairs.csv",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "artifacts" / "leakage_audit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Candidate CSV does not exist: {args.input}")

    with args.input.open(newline="", encoding="utf-8") as input_file:
        candidates = list(csv.DictReader(input_file))

    refined = []
    total = len(candidates)
    for index, candidate in enumerate(candidates, start=1):
        if index == 1 or index % 500 == 0 or index == total:
            print(f"Refining pair {index:,}/{total:,}")
        refined.append(refine_row(candidate))

    refined.sort(
        key=lambda row: (
            -row["structural_similarity"],
            row["normalized_rmse"],
        )
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_directory / "near_duplicate_pairs_refined.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(refined)

    thresholds = (0.90, 0.95, 0.98, 0.99, 0.995, 0.999)
    report = {
        "candidate_pair_count": total,
        "similarity_thresholds": {
            str(threshold): threshold_summary(refined, threshold)
            for threshold in thresholds
        },
        "interpretation": (
            "SSIM is a screening measure, not patient identification. Review the "
            "highest-scoring pairs before choosing a grouping threshold."
        ),
    }
    report_path = args.output_directory / "near_duplicate_refinement.json"
    with report_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)

    print(json.dumps(report["similarity_thresholds"], indent=2))
    print(f"Saved refined pairs to {csv_path}")
    print(f"Saved refinement summary to {report_path}")


if __name__ == "__main__":
    main()
