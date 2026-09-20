import numpy as np

from reports.refine_near_duplicates import normalized_rmse, structural_similarity


def test_identical_images_have_perfect_similarity():
    image = np.arange(128 * 128, dtype=np.float64).reshape(128, 128) % 256

    assert abs(structural_similarity(image, image) - 1.0) < 1e-12
    assert normalized_rmse(image, image) == 0.0


def test_different_images_are_less_similar():
    dark = np.zeros((128, 128), dtype=np.float64)
    bright = np.full((128, 128), 255.0, dtype=np.float64)

    assert structural_similarity(dark, bright) < 0.01
    assert normalized_rmse(dark, bright) == 1.0
