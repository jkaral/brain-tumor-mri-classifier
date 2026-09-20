"""Build leakage-resistant dataset splits without deleting source images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
CATEGORIES = ("glioma", "meningioma", "pituitary", "notumor")
LABELS = {"glioma": 1, "meningioma": 1, "pituitary": 1, "notumor": 0}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MANIFEST_FIELDS = [
    "path",
    "relative_path",
    "category",
    "binary_label",
    "source_folder",
    "assigned_split",
    "duplicate_group",
    "exclusion_reason",
]


class UnionFind:
    """Track connected duplicate and near-duplicate image groups."""

    def __init__(self, items: list[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def discover_images(dataset_root: Path) -> dict[str, dict]:
    records = {}
    for source_folder in ("Training", "Testing"):
        for category in CATEGORIES:
            category_path = dataset_root / source_folder / category
            if not category_path.is_dir():
                raise FileNotFoundError(f"Missing dataset folder: {category_path}")
            for path in sorted(category_path.iterdir()):
                if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
                    continue
                resolved = str(path.resolve())
                records[resolved] = {
                    "path": resolved,
                    "relative_path": str(path.relative_to(dataset_root)),
                    "category": category,
                    "binary_label": LABELS[category],
                    "source_folder": source_folder,
                }
    return records


def union_exact_duplicates(
    union_find: UnionFind,
    records: dict[str, dict],
    audit: dict,
) -> int:
    edge_count = 0
    for group in audit["exact_cross_split_groups"]:
        paths = [str(Path(image["path"]).resolve()) for image in group["images"]]
        paths = [path for path in paths if path in records]
        for path in paths[1:]:
            union_find.union(paths[0], path)
            edge_count += 1
    return edge_count


def union_structural_duplicates(
    union_find: UnionFind,
    records: dict[str, dict],
    refined_csv: Path,
    similarity_threshold: float,
) -> int:
    edge_count = 0
    with refined_csv.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            if float(row["structural_similarity"]) < similarity_threshold:
                continue
            left = str(Path(row["left_path"]).resolve())
            right = str(Path(row["right_path"]).resolve())
            if left not in records or right not in records:
                continue
            if records[left]["category"] != records[right]["category"]:
                raise ValueError(
                    "A high-similarity pair has conflicting source categories: "
                    f"{left} and {right}"
                )
            union_find.union(left, right)
            edge_count += 1
    return edge_count


def component_identifier(paths: list[str], dataset_root: Path) -> str:
    relative_paths = sorted(str(Path(path).relative_to(dataset_root)) for path in paths)
    digest = hashlib.sha256("\n".join(relative_paths).encode()).hexdigest()[:12]
    return f"group-{digest}"


def choose_validation_groups(
    groups: list[list[str]],
    target_images: int,
    seed: int,
) -> set[int]:
    """Choose whole groups with a subset sum nearest the requested image count."""
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    maximum_group_size = max((len(group) for group in groups), default=1)
    maximum_sum = target_images + maximum_group_size
    predecessor: dict[int, tuple[int, int] | None] = {0: None}

    for group_index in order:
        group_size = len(groups[group_index])
        for current_sum in sorted(predecessor, reverse=True):
            new_sum = current_sum + group_size
            if new_sum <= maximum_sum and new_sum not in predecessor:
                predecessor[new_sum] = (current_sum, group_index)

    best_sum = min(predecessor, key=lambda total: (abs(total - target_images), total))
    selected = set()
    while best_sum:
        previous_sum, group_index = predecessor[best_sum]
        selected.add(group_index)
        best_sum = previous_sum
    return selected


def build_assignments(
    records: dict[str, dict],
    union_find: UnionFind,
    dataset_root: Path,
    validation_fraction: float,
) -> tuple[list[dict], dict]:
    components: dict[str, list[str]] = defaultdict(list)
    for path in records:
        components[union_find.find(path)].append(path)

    assignments = {}
    group_ids = {}
    excluded_training_images = 0
    training_components_by_category: dict[str, list[list[str]]] = defaultdict(list)

    for paths in components.values():
        group_id = component_identifier(paths, dataset_root)
        for path in paths:
            group_ids[path] = group_id

        categories = {records[path]["category"] for path in paths}
        if len(categories) != 1:
            raise ValueError(f"Duplicate group crosses source categories: {paths}")

        contains_test = any(
            records[path]["source_folder"] == "Testing" for path in paths
        )
        if contains_test:
            for path in paths:
                if records[path]["source_folder"] == "Testing":
                    assignments[path] = ("test", "")
                else:
                    assignments[path] = (
                        "excluded",
                        "near_duplicate_of_test_image",
                    )
                    excluded_training_images += 1
            continue

        category = next(iter(categories))
        training_components_by_category[category].append(paths)

    validation_counts = {}
    for category_index, category in enumerate(CATEGORIES):
        groups = training_components_by_category[category]
        image_count = sum(len(group) for group in groups)
        target = round(validation_fraction * image_count)
        selected = choose_validation_groups(
            groups,
            target_images=target,
            seed=SEED + category_index,
        )
        validation_counts[category] = sum(len(groups[index]) for index in selected)

        for index, paths in enumerate(groups):
            split = "validation" if index in selected else "train"
            for path in paths:
                assignments[path] = (split, "")

    rows = []
    for path, record in records.items():
        assigned_split, exclusion_reason = assignments[path]
        rows.append(
            {
                **record,
                "assigned_split": assigned_split,
                "duplicate_group": group_ids[path],
                "exclusion_reason": exclusion_reason,
            }
        )
    rows.sort(key=lambda row: (row["assigned_split"], row["relative_path"]))

    split_counts = Counter(row["assigned_split"] for row in rows)
    category_counts = Counter(
        (row["assigned_split"], row["category"]) for row in rows
    )
    summary = {
        "split_counts": dict(split_counts),
        "category_counts": {
            f"{split}|{category}": count
            for (split, category), count in sorted(category_counts.items())
        },
        "validation_counts": validation_counts,
        "excluded_training_images": excluded_training_images,
        "component_count": len(components),
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a leakage-resistant dataset split manifest."
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "artifacts" / "leakage_audit" / "leakage_audit.json",
    )
    parser.add_argument(
        "--refined-pairs",
        type=Path,
        default=(
            ROOT
            / "artifacts"
            / "leakage_audit"
            / "near_duplicate_pairs_refined.csv"
        ),
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.995)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "artifacts" / "clean_split",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.similarity_threshold <= 1.0:
        raise ValueError("Similarity threshold must be in (0, 1].")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("Validation fraction must be in (0, 1).")

    with args.audit.open(encoding="utf-8") as input_file:
        audit = json.load(input_file)
    dataset_root = Path(audit["dataset_root"]).resolve()
    records = discover_images(dataset_root)
    union_find = UnionFind(list(records))
    exact_edges = union_exact_duplicates(union_find, records, audit)
    structural_edges = union_structural_duplicates(
        union_find,
        records,
        args.refined_pairs,
        args.similarity_threshold,
    )

    rows, summary = build_assignments(
        records,
        union_find,
        dataset_root,
        args.validation_fraction,
    )
    summary.update(
        {
            "dataset_root": str(dataset_root),
            "seed": SEED,
            "validation_fraction": args.validation_fraction,
            "structural_similarity_threshold": args.similarity_threshold,
            "exact_duplicate_edges": exact_edges,
            "structural_duplicate_edges": structural_edges,
            "policy": (
                "The original Testing folder remains test-only. Training-folder "
                "images grouped with a test image are excluded. Other duplicate "
                "groups are assigned wholly to train or validation."
            ),
        }
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_directory / "clean_split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.output_directory / "clean_split_summary.json"
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved clean manifest to {manifest_path}")
    print(f"Saved split summary to {summary_path}")


if __name__ == "__main__":
    main()
