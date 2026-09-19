from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

MODEL_URL = (
    "https://github.com/jkaral/brain-tumor-mri-classifier/"
    "releases/download/v1.0.0/"
    "best_vgg16_tumor_classifier_portable.keras"
)

EXPECTED_SHA256 = (
    "57c5dc88bc777b0cc83c7eee6293d8da"
    "0856d1208395eb95899449dd017c6dac"
)


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def ensure_model(model_path: str | Path) -> Path:
    """Return a verified local model, downloading it when necessary."""
    path = Path(model_path)

    if path.is_file():
        if calculate_sha256(path) != EXPECTED_SHA256:
            raise RuntimeError("Existing model failed SHA-256 verification.")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".part")

    request = Request(
        MODEL_URL,
        headers={"User-Agent": "brain-tumor-mri-classifier/1.0"},
    )

    try:
        with urlopen(request, timeout=180) as response:
            with temporary_path.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)

        if calculate_sha256(temporary_path) != EXPECTED_SHA256:
            raise RuntimeError("Downloaded model failed SHA-256 verification.")

        temporary_path.replace(path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return path
