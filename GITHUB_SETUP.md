# Publish this project on GitHub

The finished ZIP contains the portable model for local use, but `.gitignore` deliberately excludes the 57 MiB model from ordinary Git history. GitHub warns on repository files above 50 MiB and recommends releases or Git LFS for large binaries.

## Before publishing

1. Confirm that `artifacts/evaluation_probabilities.npz` and all three figures are present.
2. Keep the dataset archive and extracted images out of Git.
3. Decide how the portable model will be distributed. A release asset, model registry or Git LFS is preferable to ordinary Git history.

## 1. Extract and inspect the project

Extract the finished ZIP into a normal folder on your computer. Confirm that `README.md`, `app.py`, `artifacts/`, `docs/`, `models/`, `notebooks/`, `reports/`, `src/` and `tests/` are present.

## 2. Test the app locally

From a terminal inside the project folder:

```bash
python -m venv .venv
```

Activate the environment, then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload one MRI image only to confirm that the interface loads and returns an experimental model score. Stop Streamlit with `Ctrl+C`.

## 3. Create the repository locally

Run these commands from inside this project folder:

```bash
git init
git add .
git commit -m "Add VGG16 brain MRI classifier project"
git branch -M main
```

Check what will be committed:

```bash
git status
```

The dataset archive, extracted images and `.keras` model should not appear because they are excluded by `.gitignore`.

## 4. Create an empty GitHub repository

Create a new repository on your GitHub account. Do not initialize it with a README, licence or `.gitignore`, because those files already exist here.

Connect the local repository, replacing the example URL with your repository URL:

```bash
git remote add origin https://github.com/YOUR_USERNAME/brain-tumor-mri-classifier.git
git push -u origin main
```

GitHub's official instructions for this workflow are available under [adding locally hosted code to GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).

## 5. Publish the model as a release asset

Create and push a version tag:

```bash
git tag v1.0.0
git push origin v1.0.0
```

On GitHub, open the repository's **Releases** section, draft a new release using tag `v1.0.0`, and attach:

```text
models/best_vgg16_tumor_classifier_portable.keras
```

Publish the release, then add its model-download link to `models/README.md` and push that small documentation update. Do not upload the dataset archive.

GitHub documents release assets and Git LFS under [managing large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

## 6. Polish the repository page

Use this repository description:

```text
VGG16 transfer-learning classifier for tumour vs no-tumour brain MRI images, with threshold calibration, error analysis and a Streamlit demo.
```

Suggested topics:

```text
machine-learning tensorflow computer-vision transfer-learning streamlit medical-imaging
```

Confirm that the README figures render, the notebook opens, and the release is visible. On your GitHub profile, use **Customize your pins** to pin this repository.

## Alternative model-file options

The model is excluded by `.gitignore` by default. Recommended options:

1. Attach the portable `.keras` file to a GitHub release and document its checksum and download location.
2. Store it in a model registry and set `BRAIN_TUMOR_MODEL_PATH` locally.
3. Use Git LFS, then adjust `.gitignore` so the intended model file can be tracked.

Never commit `archive.zip`, extracted dataset images, Drive credentials, tokens or Streamlit secrets.
