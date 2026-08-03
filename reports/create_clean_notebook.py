"""Create the publication-ready training and evaluation notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "brain_tumor_mri_classifier_training_evaluation.ipynb"


def markdown(cell_id: str, source: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source}


def code(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


cells = [
    markdown(
        "title",
        """# Brain Tumour MRI Classifier: Training and Evaluation

This notebook documents the reproducible pipeline used for a binary tumour-versus-no-tumour MRI classifier. Images originated from [Masoud Nickparvar's Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

**Educational use only:** this model is not a medical device, has not been clinically validated, and must not be used to make healthcare decisions.

The committed model and evaluation artifacts correspond to the reported run. Retraining may produce slightly different results because of hardware and stochastic optimization.""",
    ),
    markdown(
        "setup-note",
        """## 1. Environment and configuration

In Google Colab, upload the working `archive.zip` before running the notebook. The archive is excluded from Git and should extract into `Training/` and `Testing/` category folders.""",
    ),
    code(
        "install",
        """# Colab setup; skip if the environment already has these packages.
%pip install -q tensorflow opencv-python numpy scikit-learn matplotlib""",
    ),
    code(
        "imports",
        """from pathlib import Path
import json
import random
import zipfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMAGE_SIZE = (128, 128)
CATEGORIES = ["glioma", "meningioma", "pituitary", "notumor"]
LABELS = {"glioma": 1, "meningioma": 1, "pituitary": 1, "notumor": 0}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}""",
    ),
    code(
        "extract",
        """ARCHIVE_PATH = Path("/content/archive.zip")
DATASET_PATH = Path("/content/brain-tumor-mri-dataset")

if not ARCHIVE_PATH.is_file():
    raise FileNotFoundError("Upload archive.zip to /content before continuing.")

if not (DATASET_PATH / "Training").is_dir():
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        archive.extractall(DATASET_PATH)

print("Dataset directory:", DATASET_PATH)""",
    ),
    code(
        "load-data",
        """def load_image_split(split_name):
    images, labels, category_names = [], [], []
    split_path = DATASET_PATH / split_name

    for category in CATEGORIES:
        category_path = split_path / category
        if not category_path.is_dir():
            raise FileNotFoundError(f"Missing folder: {category_path}")

        for image_path in sorted(category_path.iterdir()):
            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            images.append(image)
            labels.append(LABELS[category])
            category_names.append(category)

    images = np.asarray(images, dtype=np.float32)[..., np.newaxis] / 255.0
    labels = np.asarray(labels, dtype=np.int32)
    category_names = np.asarray(category_names)
    return images, labels, category_names


X_full_train, y_full_train, _ = load_image_split("Training")
X_test, y_test, test_categories = load_image_split("Testing")

X_train, X_val, y_train, y_val = train_test_split(
    X_full_train,
    y_full_train,
    test_size=0.20,
    random_state=SEED,
    stratify=y_full_train,
)

print("Training:", X_train.shape, np.bincount(y_train))
print("Validation:", X_val.shape, np.bincount(y_val))
print("Testing:", X_test.shape, np.bincount(y_test))""",
    ),
    markdown(
        "model-heading",
        """## 2. VGG16 transfer-learning model

The final classifier uses a frozen ImageNet-pretrained VGG16 backbone. Grayscale conversion, VGG16-compatible preprocessing and augmentation are contained in the model pipeline. The final system performs classification only; it does not segment or localize tumours.""",
    ),
    code(
        "build-model",
        """base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3),
)
base_model.trainable = False

inputs = layers.Input(shape=(128, 128, 1), name="mri_image")

x = layers.RandomRotation(0.05, fill_mode="reflect", name="random_rotation")(inputs)
x = layers.RandomZoom(0.05, 0.05, fill_mode="reflect", name="random_zoom")(x)
x = layers.RandomContrast(0.10, name="random_contrast")(x)
x = layers.Concatenate(name="grayscale_to_rgb")([x, x, x])
x = layers.Rescaling(255.0, name="convert_to_255_range")(x)
x = layers.Normalization(
    axis=-1,
    mean=[103.939, 116.779, 123.680],
    variance=[1.0, 1.0, 1.0],
    name="vgg16_normalization",
)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
x = layers.Dense(
    128,
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    name="dense_128",
)(x)
x = layers.Dropout(0.5, name="dropout")(x)
outputs = layers.Dense(1, activation="sigmoid", name="tumor_probability")(x)

model = Model(inputs, outputs, name="vgg16_brain_tumor_classifier")
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

model.summary()""",
    ),
    code(
        "train",
        """class_weights = {0: 2.0, 1: 0.6667}

callbacks = [
    EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
    ModelCheckpoint(
        "best_vgg16_tumor_classifier_portable.keras",
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)""",
    ),
    code(
        "training-curves",
        """figure, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["loss"], label="Training")
axes[0].plot(history.history["val_loss"], label="Validation")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["auc"], label="Training")
axes[1].plot(history.history["val_auc"], label="Validation")
axes[1].set_title("ROC-AUC")
axes[1].set_xlabel("Epoch")
axes[1].legend()
plt.tight_layout()
plt.show()""",
    ),
    markdown(
        "evaluation-heading",
        """## 3. Held-out evaluation and threshold calibration

The test folder remains untouched until the final evaluation. The operating threshold is selected from validation predictions only, targeting at least 97% validation recall while minimizing false positives.""",
    ),
    code(
        "probabilities",
        """validation_probabilities = model.predict(X_val, batch_size=32, verbose=1).ravel()
test_probabilities = model.predict(X_test, batch_size=32, verbose=1).ravel()

print("Exact validation ROC-AUC:", roc_auc_score(y_val, validation_probabilities))
print("Exact test ROC-AUC:", roc_auc_score(y_test, test_probabilities))""",
    ),
    code(
        "threshold",
        """TARGET_VALIDATION_RECALL = 0.97

false_positive_rates, true_positive_rates, thresholds = roc_curve(
    y_val, validation_probabilities
)
eligible = np.where(
    (true_positive_rates >= TARGET_VALIDATION_RECALL) & np.isfinite(thresholds)
)[0]
best_index = eligible[np.argmin(false_positive_rates[eligible])]
selected_threshold = float(thresholds[best_index])


def metric_summary(labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions)),
        "recall": float(recall_score(labels, predictions)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(labels, predictions)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


validation_metrics = metric_summary(
    y_val, validation_probabilities, selected_threshold
)
test_metrics = metric_summary(y_test, test_probabilities, selected_threshold)

print("Selected threshold:", selected_threshold)
print("Validation metrics:", validation_metrics)
print("Test metrics:", test_metrics)

test_predictions = (test_probabilities >= selected_threshold).astype(np.int32)
ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_predictions,
    display_labels=["No tumour", "Tumour"],
    cmap="Blues",
)
plt.title("Held-out test confusion matrix")
plt.show()""",
    ),
    code(
        "subtypes",
        """print("Performance by source category")
for category in CATEGORIES:
    category_mask = test_categories == category
    expected_label = LABELS[category]
    category_predictions = test_predictions[category_mask]
    correct = int(np.sum(category_predictions == expected_label))
    total = int(np.sum(category_mask))
    print(f"{category}: {correct}/{total} correct ({correct / total:.4f})")""",
    ),
    code(
        "save-artifacts",
        """ARTIFACT_DIRECTORY = Path("artifacts")
ARTIFACT_DIRECTORY.mkdir(exist_ok=True)

np.savez_compressed(
    ARTIFACT_DIRECTORY / "evaluation_probabilities.npz",
    y_val=y_val,
    validation_probabilities=validation_probabilities,
    y_test=y_test,
    test_probabilities=test_probabilities,
    test_categories=test_categories,
)

report = {
    "selected_threshold": selected_threshold,
    "validation_roc_auc": float(roc_auc_score(y_val, validation_probabilities)),
    "test_roc_auc": float(roc_auc_score(y_test, test_probabilities)),
    "validation_metrics": validation_metrics,
    "test_metrics": test_metrics,
}

with (ARTIFACT_DIRECTORY / "threshold_evaluation.json").open("w") as output_file:
    json.dump(report, output_file, indent=2)

model.save("best_vgg16_tumor_classifier_portable.keras")""",
    ),
    markdown(
        "reported-results",
        """## 4. Reported run

The repository's committed artifacts produced the following held-out test results at threshold `0.287156`:

- accuracy: **94.69%**
- exact ROC-AUC from saved scores: **0.9963**
- precision: **99.82%**
- recall: **93.08%**
- specificity: **99.50%**
- F1-score: **96.33%**
- confusion matrix: TN 398, FP 2, FN 83, TP 1,117

Glioma and meningioma accounted for 77 of the 83 remaining false negatives. Performance was measured on one dataset with image-level labels and no verified patient-level split; related images across splits could inflate results. The sigmoid output is a model score, not a calibrated medical probability.""",
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Created {OUTPUT}")

