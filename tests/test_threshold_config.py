from pathlib import Path

import pytest

from src.inference import load_threshold


def test_saved_threshold():
    project_root = Path(__file__).resolve().parents[1]
    threshold = load_threshold(
        project_root / "artifacts" / "threshold_evaluation.json"
    )
    assert threshold == pytest.approx(0.287156)

