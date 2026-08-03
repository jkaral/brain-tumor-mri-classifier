# Resume and interview wording

## Recommended project title

**Brain Tumour MRI Classifier** — Python, TensorFlow, Keras, OpenCV, scikit-learn

## Resume bullets

- Developed a VGG16 transfer-learning pipeline to classify tumour versus no-tumour brain MRI images across a 7,200-image dataset, incorporating augmentation, class weighting, early stopping and held-out evaluation.
- Achieved **94.7% test accuracy, 0.9963 ROC-AUC, 99.8% precision and 93.1% recall** on 1,600 held-out images; reduced false negatives by **23.9%** through validation-based threshold calibration and conducted tumour-category error analysis.

After the Streamlit demo is running, optionally add:

- Built an interactive Streamlit interface for reproducible image preprocessing and model inference, with documented failure modes and responsible-use limitations.

## Thirty-second interview explanation

I built a binary brain-MRI classifier using a frozen VGG16 backbone and a custom classification head. I separated training, validation and test data, used validation ROC-AUC for checkpoint selection, and initially obtained 93.1% test accuracy with very high precision but 109 false negatives. I then selected a lower threshold using validation data only, reducing false negatives by 23.9% and raising test accuracy to 94.7%. My error analysis showed that glioma and meningioma accounted for most remaining misses. The project is an educational classifier, not a clinical system, and I documented dataset and patient-level-splitting limitations.

## Claims to avoid

- “Diagnoses brain cancer”
- “Clinically accurate” or “medical-grade”
- “Segments” or “localizes” tumours
- “99% accurate”
- “Works on CT scans”
- “Predicts tumour subtype”
