import numpy as np
from PIL import Image

from src.preprocessing import preprocess_image


def test_preprocess_image_shape_dtype_and_range():
    source = np.arange(24 * 32, dtype=np.uint8).reshape(24, 32)
    image = Image.fromarray(source, mode="L")

    processed = preprocess_image(image)

    assert processed.shape == (1, 128, 128, 1)
    assert processed.dtype == np.float32
    assert 0.0 <= float(processed.min())
    assert float(processed.max()) <= 1.0

