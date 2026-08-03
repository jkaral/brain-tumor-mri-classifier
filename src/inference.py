"""Model loading, configuration, and inference helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from src.preprocessing import preprocess_image


def load_threshold(config_path: str | Path) -> float:
    """Load and validate the decision threshold saved during evaluation."""
    path = Path(config_path)
    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    threshold = float(config["selected_threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError("The configured decision threshold must be between 0 and 1.")
    return threshold


def load_keras_model(model_path: str | Path) -> Any:
    """Load the portable Keras model lazily so utility imports stay lightweight."""
    import tensorflow as tf

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found at {path}. See models/README.md for setup instructions."
        )
    return tf.keras.models.load_model(path, compile=False)


def predict_score(model: Any, image: Image.Image) -> float:
    """Return the model's scalar tumour score for one image."""
    model_input = preprocess_image(image)
    raw_prediction = model.predict(model_input, verbose=0)
    score = float(raw_prediction.reshape(-1)[0])
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"Model returned an invalid score: {score}")
    return score

