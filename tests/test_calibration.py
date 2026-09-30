import numpy as np

from reports.evaluate_calibration import (
    apply_temperature,
    calibrated_threshold,
    expected_calibration_error,
    fit_temperature,
    validation_confidence_cutoff,
)


def test_temperature_one_preserves_probabilities():
    probabilities = np.array([0.1, 0.3, 0.7, 0.9])

    scaled = apply_temperature(probabilities, 1.0)

    np.testing.assert_allclose(
        scaled,
        probabilities,
        atol=1e-7,
    )


def test_temperature_scaling_preserves_probability_order():
    probabilities = np.array([0.05, 0.2, 0.6, 0.95])

    scaled = apply_temperature(probabilities, 2.0)

    assert np.all(np.diff(scaled) > 0)


def test_calibrated_threshold_preserves_decisions():
    probabilities = np.array([0.1, 0.25, 0.4, 0.8, 0.95])
    threshold = 0.35
    temperature = 1.8

    original_predictions = probabilities >= threshold

    scaled_probabilities = apply_temperature(
        probabilities,
        temperature,
    )
    scaled_threshold = calibrated_threshold(
        threshold,
        temperature,
    )

    scaled_predictions = scaled_probabilities >= scaled_threshold

    np.testing.assert_array_equal(
        original_predictions,
        scaled_predictions,
    )


def test_expected_calibration_error_is_zero_for_perfect_groups():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])

    ece = expected_calibration_error(
        labels,
        probabilities,
        bins=2,
    )

    assert ece == 0.0


def test_fit_temperature_returns_positive_value():
    labels = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array(
        [0.10, 0.20, 0.35, 0.65, 0.80, 0.90]
    )

    temperature = fit_temperature(labels, probabilities)

    assert temperature > 0


def test_validation_confidence_cutoff_is_nonnegative():
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    cutoff = validation_confidence_cutoff(
        probabilities,
        decision_threshold=0.5,
        target_coverage=0.5,
    )

    assert cutoff >= 0
