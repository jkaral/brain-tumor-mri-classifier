"""Retrain and evaluate the original VGG16 baseline on leakage-resistant splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent
SEED = 42
IMAGE_SIZE = (128, 128)


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    category: str
    split: str


def read_manifest(
    manifest_path: Path,
    dataset_root: Path,
) -> dict[str, list[ImageRecord]]:
    """Resolve portable manifest paths against the selected dataset root."""
    records: dict[str, list[ImageRecord]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    with manifest_path.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            split = row["assigned_split"]
            if split not in records:
                continue
            path = dataset_root / row["relative_path"]
            if not path.is_file():
                raise FileNotFoundError(f"Manifest image does not exist: {path}")
            records[split].append(
                ImageRecord(
                    path=path,
                    label=int(row["binary_label"]),
                    category=row["category"],
                    split=split,
                )
            )
    if any(not split_records for split_records in records.values()):
        raise ValueError("Manifest must contain train, validation, and test images.")
    return records


def balanced_class_weights(records: list[ImageRecord]) -> dict[int, float]:
    counts = Counter(record.label for record in records)
    if set(counts) != {0, 1}:
        raise ValueError("Training split must contain both binary classes.")
    total = len(records)
    return {label: total / (2.0 * count) for label, count in counts.items()}


def metric_summary(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    predictions = (probabilities >= threshold).astype(np.int32)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(true_negative / (true_negative + false_positive)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "true_negatives": int(true_negative),
        "false_positives": int(false_positive),
        "false_negatives": int(false_negative),
        "true_positives": int(true_positive),
    }


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_recall: float,
) -> float:
    false_positive_rates, true_positive_rates, thresholds = roc_curve(
        labels,
        probabilities,
    )
    eligible = np.where(
        (true_positive_rates >= target_recall) & np.isfinite(thresholds)
    )[0]
    if eligible.size == 0:
        raise ValueError("No finite threshold reaches the requested validation recall.")
    best_index = eligible[np.argmin(false_positive_rates[eligible])]
    return float(thresholds[best_index])


def make_dataset(records: list[ImageRecord], batch_size: int, training: bool):
    import tensorflow as tf

    paths = [str(record.path) for record in records]
    labels = np.asarray([record.label for record in records], dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, label):
        encoded = tf.io.read_file(path)
        image = tf.io.decode_image(
            encoded,
            channels=1,
            expand_animations=False,
        )
        image.set_shape((None, None, 1))
        image = tf.image.resize(image, IMAGE_SIZE, method="area")
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not training,
    )
    if training:
        dataset = dataset.shuffle(
            len(records),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model():
    import tensorflow as tf

    base_model = tf.keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(128, 128, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=(128, 128, 1), name="mri_image")
    x = tf.keras.layers.RandomRotation(
        0.05,
        fill_mode="reflect",
        name="random_rotation",
    )(inputs)
    x = tf.keras.layers.RandomZoom(
        0.05,
        0.05,
        fill_mode="reflect",
        name="random_zoom",
    )(x)
    x = tf.keras.layers.RandomContrast(0.10, name="random_contrast")(x)
    x = tf.keras.layers.Concatenate(name="grayscale_to_rgb")([x, x, x])
    x = tf.keras.layers.Rescaling(255.0, name="convert_to_255_range")(x)
    x = tf.keras.layers.Normalization(
        axis=-1,
        mean=[103.939, 116.779, 123.680],
        variance=[1.0, 1.0, 1.0],
        name="vgg16_normalization",
    )(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="dense_128",
    )(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="tumor_probability",
    )(x)
    model = tf.keras.Model(inputs, outputs, name="vgg16_clean_baseline")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def subtype_results(
    records: list[ImageRecord],
    probabilities: np.ndarray,
    threshold: float,
) -> list[dict]:
    predictions = (probabilities >= threshold).astype(np.int32)
    categories = np.asarray([record.category for record in records])
    labels = np.asarray([record.label for record in records])
    results = []
    for category in ("glioma", "meningioma", "pituitary", "notumor"):
        mask = categories == category
        correct = int(np.sum(predictions[mask] == labels[mask]))
        total = int(np.sum(mask))
        results.append(
            {
                "category": category,
                "images": total,
                "correct": correct,
                "errors": total - correct,
                "accuracy": correct / total,
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the original frozen-VGG16 baseline on clean splits."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "artifacts" / "clean_split" / "clean_split_manifest.csv",
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-validation-recall", type=float, default=0.97)
    parser.add_argument(
        "--output-model",
        type=Path,
        default=ROOT / "models" / "best_vgg16_clean_baseline.keras",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "artifacts" / "clean_baseline",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_root is None:
        summary_path = args.manifest.parent / "clean_split_summary.json"
        with summary_path.open(encoding="utf-8") as input_file:
            args.dataset_root = Path(json.load(input_file)["dataset_root"])

    import tensorflow as tf

    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    records = read_manifest(args.manifest, args.dataset_root.resolve())
    train_dataset = make_dataset(records["train"], args.batch_size, training=True)
    validation_dataset = make_dataset(
        records["validation"],
        args.batch_size,
        training=False,
    )
    test_dataset = make_dataset(records["test"], args.batch_size, training=False)
    class_weights = balanced_class_weights(records["train"])

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    model = build_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            args.output_model,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    validation_probabilities = model.predict(validation_dataset, verbose=1).ravel()
    test_probabilities = model.predict(test_dataset, verbose=1).ravel()
    validation_labels = np.asarray(
        [record.label for record in records["validation"]],
        dtype=np.int32,
    )
    test_labels = np.asarray(
        [record.label for record in records["test"]],
        dtype=np.int32,
    )
    threshold = select_threshold(
        validation_labels,
        validation_probabilities,
        args.target_validation_recall,
    )

    report = {
        "seed": SEED,
        "model": "VGG16 frozen baseline",
        "image_size": list(IMAGE_SIZE),
        "class_weights": {str(key): value for key, value in class_weights.items()},
        "split_counts": {key: len(value) for key, value in records.items()},
        "target_validation_recall": args.target_validation_recall,
        "selected_threshold": threshold,
        "validation_roc_auc": float(
            roc_auc_score(validation_labels, validation_probabilities)
        ),
        "test_roc_auc": float(roc_auc_score(test_labels, test_probabilities)),
        "validation_metrics": metric_summary(
            validation_labels,
            validation_probabilities,
            threshold,
        ),
        "default_test_metrics": metric_summary(
            test_labels,
            test_probabilities,
            0.5,
        ),
        "tuned_test_metrics": metric_summary(
            test_labels,
            test_probabilities,
            threshold,
        ),
        "tuned_subtype_results": subtype_results(
            records["test"],
            test_probabilities,
            threshold,
        ),
        "epochs_completed": len(history.history["loss"]),
        "history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
    }

    args.output_directory.mkdir(parents=True, exist_ok=True)
    report_path = args.output_directory / "evaluation.json"
    with report_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)
    np.savez_compressed(
        args.output_directory / "probabilities.npz",
        y_val=validation_labels,
        validation_probabilities=validation_probabilities,
        y_test=test_labels,
        test_probabilities=test_probabilities,
        test_categories=np.asarray(
            [record.category for record in records["test"]]
        ),
    )

    printable_report = {
        key: value for key, value in report.items() if key != "history"
    }
    print(json.dumps(printable_report, indent=2))
    print(f"Saved model to {args.output_model}")
    print(f"Saved evaluation to {report_path}")


if __name__ == "__main__":
    main()
