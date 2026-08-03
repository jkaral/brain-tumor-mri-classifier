# Model Card: VGG16 Brain Tumour MRI Classifier

## Model details

- **Task:** binary MRI image classification
- **Classes:** tumour and no tumour
- **Backbone:** VGG16 pretrained on ImageNet
- **Input:** one grayscale image resized to 128 × 128 pixels
- **Output:** sigmoid model score between zero and one
- **Decision threshold:** 0.287156, selected on validation data
- **Framework:** TensorFlow/Keras

## Intended use

This model is an educational machine-learning demonstration and portfolio project. It is intended to demonstrate transfer learning, evaluation, threshold calibration, error analysis, reproducibility, and responsible communication of limitations.

## Out-of-scope use

The model must not be used for medical diagnosis, screening, triage, treatment selection, patient reassurance, or any other healthcare decision. It must not be presented as clinically validated. Its output must not replace interpretation by qualified healthcare professionals.

## Evaluation

On the 1,600-image held-out test folder, the locked threshold produced:

- accuracy: 94.69%
- exact score-level ROC-AUC: 0.9963
- precision: 99.82%
- recall: 93.08%
- specificity: 99.50%
- F1-score: 96.33%
- confusion matrix: TN 398, FP 2, FN 83, TP 1,117

Performance by source category was 89.25% for glioma, 91.50% for meningioma, 98.50% for pituitary and 99.50% for no-tumour images.

## Threshold selection

The default 0.5 threshold produced 109 false negatives. A threshold of 0.287156 was selected using validation data to achieve at least 97% validation recall while minimizing false positives. On the test set, it reduced false negatives to 83 and increased false positives from one to two. The test set was not used to select the threshold.

## Major limitations

1. Evaluation used one dataset without external replication.
2. Patient identifiers were unavailable, so patient-level separation could not be verified.
3. Image-level correlations or duplicated/augmented source images may inflate results.
4. Dataset class proportions do not represent clinical prevalence.
5. The model does not localize abnormalities or determine tumour subtype.
6. The score is not calibrated as an individual medical probability.
7. Glioma and meningioma accounted for most false negatives.
8. Robustness across scanners, hospitals, acquisition sequences and populations is unknown.

## Training-data note

The model was trained from image-level class labels. KMeans pseudo-masks created during an early experiment were not used by the final classifier and are not ground-truth tumour annotations.

## Dataset source

The working images originated from Masoud Nickparvar's [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). This repository does not redistribute the images. Users must review the source page's licence and usage terms. Patient-level structure and collection details were not available in the working archive and remain limitations.
