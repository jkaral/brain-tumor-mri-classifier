"""Image preprocessing shared by the app and tests."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = (128, 128)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert an uploaded image to the model's normalized grayscale tensor."""
    grayscale = np.asarray(ImageOps.grayscale(image), dtype=np.uint8)
    resized = cv2.resize(grayscale, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    return image_array[np.newaxis, ..., np.newaxis]
