"""Streamlit interface for the experimental MRI classifier."""

from __future__ import annotations
from src.model_download import ensure_model

import os
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.inference import load_keras_model, load_threshold, predict_score

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "models" / "best_vgg16_tumor_classifier_portable.keras"
MODEL_PATH = Path(os.getenv("BRAIN_TUMOR_MODEL_PATH", DEFAULT_MODEL_PATH))
CONFIG_PATH = ROOT / "artifacts" / "threshold_evaluation.json"


@st.cache_resource
def get_model(model_path: str):
    return load_keras_model(model_path)


st.set_page_config(
    page_title="Brain Tumour MRI Classifier",
    page_icon="🧠",
    layout="centered",
)

st.title("Brain Tumour MRI Classifier")
st.caption("VGG16 transfer-learning demonstration · Binary MRI classification")

st.warning(
    "Educational demonstration only. This system is not a medical device, does not "
    "provide a diagnosis, and must not be used to make healthcare decisions."
)

threshold = load_threshold(CONFIG_PATH)

with st.sidebar:
    st.header("Held-out test results")
    st.metric("ROC-AUC", "0.9963")
    st.metric("Accuracy", "94.69%")
    st.metric("Recall", "93.08%")
    st.metric("Specificity", "99.50%")
    st.caption(f"Validation-selected threshold: {threshold:.6f}")

uploaded_file = st.file_uploader(
    "Upload one brain MRI image",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Accepted formats: JPG, JPEG, PNG, and BMP.",
)

if uploaded_file is None:
    st.info("Upload an image to run the experimental classifier.")
else:
    try:
        uploaded_image = Image.open(uploaded_file)
        uploaded_image.load()
    except (UnidentifiedImageError, OSError):
        st.error("The uploaded file could not be read as an image.")
        st.stop()

    st.image(uploaded_image, caption="Uploaded image", use_container_width=True)

        if st.button("Run classifier", type="primary", use_container_width=True):
        try:
            with st.spinner("Preparing model..."):
                verified_model_path = ensure_model(MODEL_PATH)
                model = get_model(str(verified_model_path))

            with st.spinner("Processing image..."):
                score = predict_score(model, uploaded_image)

        except FileNotFoundError as error:
            st.error(str(error))
            st.stop()

        except Exception:
            st.error(
                "Inference failed. Confirm that the portable model and compatible "
                "TensorFlow version are installed."
            )
            st.stop()

        st.subheader("Experimental output")
        st.progress(score)
        st.write(f"Model score: **{score:.4f}**")

        if score >= threshold:
            st.warning("Classification: tumour-like pattern detected")
        else:
            st.success("Classification: no-tumour pattern detected")

        st.caption(
            "The model score is not a calibrated medical probability. False negatives "
            "and false positives remain possible."
        )

with st.spinner("Processing image..."):
    score = predict_score(model, uploaded_image)
                score = predict_score(model, uploaded_image)
        except FileNotFoundError as error:
            st.error(str(error))
            st.stop()
        except Exception:
            st.error(
                "Inference failed. Confirm that the portable model and compatible "
                "TensorFlow version are installed."
            )
            st.stop()

        st.subheader("Experimental output")
        st.progress(score)
        st.write(f"Model score: **{score:.4f}**")

        if score >= threshold:
            st.warning("Classification: tumour-like pattern detected")
        else:
            st.success("Classification: no-tumour pattern detected")

        st.caption(
            "The model score is not a calibrated medical probability. False negatives "
            "and false positives remain possible."
        )
