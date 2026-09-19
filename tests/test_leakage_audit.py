from pathlib import Path

import numpy as np
from PIL import Image

from reports.audit_dataset_leakage import (
    ImageRecord,
    difference_hash,
    exact_cross_split_groups,
    hamming_distance,
)


def test_difference_hash_is_stable_for_identical_images(tmp_path: Path):
    pixels = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.fromarray(pixels, mode="L").save(first)
    Image.fromarray(pixels, mode="L").save(second)

    assert difference_hash(first) == difference_hash(second)


def test_hamming_distance_counts_different_bits():
    assert hamming_distance(0b1010, 0b0011) == 2


def test_exact_duplicates_are_reported_only_across_splits():
    shared = "a" * 64
    records = [
        ImageRecord("train", "glioma", Path("a.png"), shared, 1),
        ImageRecord("test", "glioma", Path("b.png"), shared, 1),
        ImageRecord("train", "notumor", Path("c.png"), "b" * 64, 2),
    ]

    groups = exact_cross_split_groups(records)

    assert len(groups) == 1
    assert {image["split"] for image in groups[0]["images"]} == {"train", "test"}
