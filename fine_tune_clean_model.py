"""Fine-tune VGG16 block 5 using the leakage-resistant split manifest."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from train_clean_baseline import (
    ROOT,
    SEED,
    balanced_class_weights,
    make_dataset,
    metric_summary,
    read_manifest,
    select_threshold,
    subtype_results,
)


def configure_block5_for_fine_tuning(model) -> list[str]:
    """Freeze VGG16 blocks 1-4 and unfreeze the final convolutional block."""
    try:
        base_model = model.get_layer("vgg16")
    except ValueError as error:
        raise ValueError(
            "The input model does not contain the expected 'vgg16' layer."
        ) from error

    base_model.trainable = True
    trainable_layers = []
    for layer in base_model.layers:
        layer.trainable = layer.name.startswith("block5_")
        if layer.trainable and layer.weights:
            trainable_layers.append(layer.name)

    if not trainable_layers:
        raise ValueError("No trainable VGG16 block-5 layers were found.")
    return trainable_layers


def compile_for_fine_tuning(model, learning_rate: float) -> None:
    """Recompile after changing trainability so Keras applies the changes."""
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune VGG16 block 5 from the clean frozen baseline."
    )
    parser.add_argument(
        "--input-model",
        type=Path,
        default=ROOT / "models" / "best_vgg16_clean_baseline.keras",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "artifacts" / "clean_split" / "clean_split_manifest.csv",
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--target-validation-recall", type=float, default=0.97)
    parser.add_argument(
        "--output-model",
        type=Path,
        default=ROOT / "models" / "best_vgg16_clean_finetuned.keras",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "artifacts" / "clean_finetuned",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_model.is_file():
        raise FileNotFoundError(
            f"Clean baseline model does not exist: {args.input_model}"
        )
    if args.learning_rate <= 0:
        raise ValueError("Learning rate must be positive.")

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
        records["validation"], args.batch_size, training=False
    )
    test_dataset = make_dataset(records["test"], args.batch_size, training=False)
    class_weights = balanced_class_weights(records["train"])

    model = tf.keras.models.load_model(args.input_model, compile=False)
    trainable_vgg_layers = configure_block5_for_fine_tuning(model)
    compile_for_fine_tuning(model, args.learning_rate)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=args.output_directory / "training_backup",
            delete_checkpoint=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.2,
            patience=1,
            min_lr=1e-7,
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

    print(f"Fine-tuning VGG16 layers: {', '.join(trainable_vgg_layers)}")
    print(f"Learning rate: {args.learning_rate:g}")
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
        [record.label for record in records["validation"]], dtype=np.int32
    )
    test_labels = np.asarray(
        [record.label for record in records["test"]], dtype=np.int32
    )
    threshold = select_threshold(
        validation_labels,
        validation_probabilities,
        args.target_validation_recall,
    )

    report = {
        "seed": SEED,
        "model": "VGG16 clean baseline with block 5 fine-tuned",
        "input_model": str(args.input_model),
        "fine_tuned_vgg_layers": trainable_vgg_layers,
        "learning_rate": args.learning_rate,
        "class_weights": {str(key): value for key, value in class_weights.items()},
        "split_counts": {key: len(value) for key, value in records.items()},
        "target_validation_recall": args.target_validation_recall,
        "selected_threshold": threshold,
        "validation_roc_auc": float(
            roc_auc_score(validation_labels, validation_probabilities)
        ),
        "test_roc_auc": float(roc_auc_score(test_labels, test_probabilities)),
        "validation_metrics": metric_summary(
            validation_labels, validation_probabilities, threshold
        ),
        "default_test_metrics": metric_summary(
            test_labels, test_probabilities, 0.5
        ),
        "tuned_test_metrics": metric_summary(
            test_labels, test_probabilities, threshold
        ),
        "tuned_subtype_results": subtype_results(
            records["test"], test_probabilities, threshold
        ),
        "epochs_completed": len(history.history["loss"]),
        "history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
    }

    baseline_report_path = ROOT / "artifacts" / "clean_baseline" / "evaluation.json"
    if baseline_report_path.is_file():
        with baseline_report_path.open(encoding="utf-8") as input_file:
            baseline = json.load(input_file)
        report["comparison_to_frozen_baseline"] = {
            "test_roc_auc_change": (
                report["test_roc_auc"] - baseline["test_roc_auc"]
            ),
            "default_accuracy_change": (
                report["default_test_metrics"]["accuracy"]
                - baseline["default_test_metrics"]["accuracy"]
            ),
            "tuned_accuracy_change": (
                report["tuned_test_metrics"]["accuracy"]
                - baseline["tuned_test_metrics"]["accuracy"]
            ),
            "tuned_recall_change": (
                report["tuned_test_metrics"]["recall"]
                - baseline["tuned_test_metrics"]["recall"]
            ),
        }

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
    print(f"Saved fine-tuned model to {args.output_model}")
    print(f"Saved evaluation to {report_path}")


if __name__ == "__main__":
    main()
