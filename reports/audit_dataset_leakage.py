"""Audit exact and near-duplicate leakage across MRI dataset splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
CATEGORIES = ("glioma", "meningioma", "pituitary", "notumor")
LABELS = {"glioma": 1, "meningioma": 1, "pituitary": 1, "notumor": 0}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class ImageRecord:
    """Hashes and split metadata for one source image."""

    split: str
    category: str
    path: Path
    exact_hash: str
    perceptual_hash: int


def sha256_file(path: Path) -> str:
    """Return a byte-level SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(path: Path, hash_size: int = 8) -> int:
    """Return a small perceptual difference hash for near-duplicate screening."""
    with Image.open(path) as image:
        grayscale = ImageOps.grayscale(image)
        resized = grayscale.resize(
            (hash_size + 1, hash_size),
            Image.Resampling.LANCZOS,
        )
        pixels = np.asarray(resized, dtype=np.int16)

    differences = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in differences.ravel():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left: int, right: int) -> int:
    """Count differing bits between two perceptual hashes."""
    return (left ^ right).bit_count()


def category_images(dataset_root: Path, folder: str, category: str) -> list[Path]:
    category_path = dataset_root / folder / category
    if not category_path.is_dir():
        raise FileNotFoundError(f"Missing dataset folder: {category_path}")
    return [
        path
        for path in sorted(category_path.iterdir())
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]


def reproduce_split_membership(dataset_root: Path) -> list[tuple[str, str, Path]]:
    """Reproduce the notebook's seed-42 binary-stratified train/validation split."""
    training_items: list[tuple[str, Path]] = []
    training_labels: list[int] = []

    for category in CATEGORIES:
        paths = category_images(dataset_root, "Training", category)
        training_items.extend((category, path) for path in paths)
        training_labels.extend([LABELS[category]] * len(paths))

    train_items, validation_items = train_test_split(
        training_items,
        test_size=0.20,
        random_state=SEED,
        stratify=training_labels,
    )

    membership = [
        ("train", category, path) for category, path in train_items
    ]
    membership.extend(
        ("validation", category, path) for category, path in validation_items
    )

    for category in CATEGORIES:
        membership.extend(
            ("test", category, path)
            for path in category_images(dataset_root, "Testing", category)
        )

    return membership


def build_records(dataset_root: Path) -> list[ImageRecord]:
    records = []
    membership = reproduce_split_membership(dataset_root)
    total = len(membership)

    for index, (split, category, path) in enumerate(membership, start=1):
        if index == 1 or index % 500 == 0 or index == total:
            print(f"Hashing image {index:,}/{total:,}")
        records.append(
            ImageRecord(
                split=split,
                category=category,
                path=path,
                exact_hash=sha256_file(path),
                perceptual_hash=difference_hash(path),
            )
        )
    return records


def exact_cross_split_groups(records: list[ImageRecord]) -> list[dict]:
    """Return byte-identical groups that occur in more than one split."""
    by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_hash[record.exact_hash].append(record)

    groups = []
    for exact_hash, matches in by_hash.items():
        if len({match.split for match in matches}) < 2:
            continue
        groups.append(
            {
                "sha256": exact_hash,
                "images": [
                    {
                        "split": match.split,
                        "category": match.category,
                        "path": str(match.path),
                    }
                    for match in matches
                ],
            }
        )
    return groups


def scan_near_duplicates(
    records: list[ImageRecord],
    maximum_distance: int,
    csv_path: Path,
    maximum_reported_pairs: int,
) -> tuple[int, int]:
    """Scan only cross-split image pairs and write a bounded review list."""
    by_split: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_split[record.split].append(record)

    total_matches = 0
    written_matches = 0
    fieldnames = [
        "hamming_distance",
        "left_split",
        "left_category",
        "left_path",
        "right_split",
        "right_category",
        "right_path",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for left_split, right_split in combinations(
            ("train", "validation", "test"), 2
        ):
            print(f"Comparing {left_split} against {right_split}")
            for left in by_split[left_split]:
                for right in by_split[right_split]:
                    if left.exact_hash == right.exact_hash:
                        continue
                    distance = hamming_distance(
                        left.perceptual_hash,
                        right.perceptual_hash,
                    )
                    if distance > maximum_distance:
                        continue

                    total_matches += 1
                    if written_matches >= maximum_reported_pairs:
                        continue

                    writer.writerow(
                        {
                            "hamming_distance": distance,
                            "left_split": left.split,
                            "left_category": left.category,
                            "left_path": left.path,
                            "right_split": right.split,
                            "right_category": right.category,
                            "right_path": right.path,
                        }
                    )
                    written_matches += 1

    return total_matches, written_matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit exact and near-duplicate images across dataset splits."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Directory containing the Training and Testing folders.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "artifacts" / "leakage_audit",
    )
    parser.add_argument(
        "--near-duplicate-distance",
        type=int,
        default=4,
        help="Maximum 64-bit dHash Hamming distance to flag for manual review.",
    )
    parser.add_argument(
        "--maximum-reported-pairs",
        type=int,
        default=10_000,
        help="Maximum near-duplicate pairs written to CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not 0 <= args.near_duplicate_distance <= 64:
        raise ValueError("Near-duplicate distance must be between 0 and 64.")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    records = build_records(dataset_root)
    exact_groups = exact_cross_split_groups(records)

    near_csv = args.output_directory / "near_duplicate_pairs.csv"
    near_count, written_count = scan_near_duplicates(
        records,
        args.near_duplicate_distance,
        near_csv,
        args.maximum_reported_pairs,
    )

    report = {
        "dataset_root": str(dataset_root),
        "split_counts": dict(Counter(record.split for record in records)),
        "near_duplicate_hamming_threshold": args.near_duplicate_distance,
        "exact_cross_split_group_count": len(exact_groups),
        "exact_cross_split_groups": exact_groups,
        "near_duplicate_cross_split_pair_count": near_count,
        "near_duplicate_pairs_written": written_count,
        "near_duplicate_pairs_truncated": near_count > written_count,
        "limitations": [
            "Perceptual-hash matches are candidates and require manual review.",
            "This audit cannot prove patient independence without patient IDs.",
            "A separate external dataset is still required for external validation.",
        ],
    }

    report_path = args.output_directory / "leakage_audit.json"
    with report_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)

    print(f"Exact cross-split duplicate groups: {len(exact_groups):,}")
    print(f"Near-duplicate cross-split candidates: {near_count:,}")
    print(f"Saved JSON report to {report_path}")
    print(f"Saved review CSV to {near_csv}")


if __name__ == "__main__":
    main()
