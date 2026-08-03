# Model file

Place the verified portable model here as:

```text
best_vgg16_tumor_classifier_portable.keras
```

Verified SHA-256 checksum:

```text
57c5dc88bc777b0cc83c7eee6293d8da0856d1208395eb95899449dd017c6dac
```

The original checkpoint containing an anonymous Lambda layer is retained only as a recovery backup. The Streamlit app should use the portable model.

The model is approximately 57 MiB. GitHub warns when ordinary repository files exceed 50 MiB, so distribute this file through a release asset, model registry or Git LFS rather than ordinary Git history. If the model is not stored here, set the environment variable `BRAIN_TUMOR_MODEL_PATH` to its absolute path before starting Streamlit.
