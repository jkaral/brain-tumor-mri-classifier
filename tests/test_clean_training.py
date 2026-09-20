import csv
from pathlib import Path

from train_clean_baseline import balanced_class_weights, read_manifest


def test_manifest_reader_ignores_excluded_images(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    paths = [
        dataset_root / "Training" / "glioma" / "train.png",
        dataset_root / "Training" / "notumor" / "validation.png",
        dataset_root / "Testing" / "glioma" / "test.png",
        dataset_root / "Training" / "glioma" / "excluded.png",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    manifest = tmp_path / "manifest.csv"
    fieldnames = ["relative_path", "binary_label", "category", "assigned_split"]
    rows = [
        ["Training/glioma/train.png", 1, "glioma", "train"],
        ["Training/notumor/validation.png", 0, "notumor", "validation"],
        ["Testing/glioma/test.png", 1, "glioma", "test"],
        ["Training/glioma/excluded.png", 1, "glioma", "excluded"],
    ]
    with manifest.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(fieldnames)
        writer.writerows(rows)

    records = read_manifest(manifest, dataset_root)

    assert len(records["train"]) == 1
    assert len(records["validation"]) == 1
    assert len(records["test"]) == 1


def test_balanced_class_weights_reflect_class_frequency():
    from train_clean_baseline import ImageRecord

    records = [
        ImageRecord(Path("a"), 0, "notumor", "train"),
        ImageRecord(Path("b"), 1, "glioma", "train"),
        ImageRecord(Path("c"), 1, "glioma", "train"),
        ImageRecord(Path("d"), 1, "glioma", "train"),
    ]

    weights = balanced_class_weights(records)

    assert weights == {0: 2.0, 1: 2.0 / 3.0}
