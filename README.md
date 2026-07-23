# Learning to Predict Behaviour from EEG Dynamics

## MSc Dissertation Project

This repository contains the code developed for my MSc dissertation at the University of Sheffield.

The aim of this project is to investigate whether behavioural patterns can be predicted from EEG recordings using machine learning. The project follows an EEG analysis pipeline including signal preprocessing, feature extraction, feature selection using SHAP, and classification using multiple machine learning models.

---

## Project Overview

The workflow of the project is:

1. Load raw EEG recordings.
2. Preprocess EEG signals.
3. Extract frequency-domain and time-domain features.
4. Perform Leave-One-Subject-Out (LOSO) evaluation.
5. Compute SHAP feature importance across participants.
6. Select the most important features.
7. Retrain classifiers using only selected features.
8. Compare classification performance using AUC-ROC.

---

## EEG Preprocessing

The EEG recordings are processed using the MNE Python package.

The preprocessing pipeline includes:

- Filtering EEG signals between 1-40 Hz.
- Selecting frontal EEG channels.
- Creating epochs based on behavioural events.
- Extracting 1-second EEG segments around events before feature extraction.

The EEG channels used are:

- Fz
- FCz
- F2
- F3
- F4
- F6

---

## Repository Structure

| File | Description |
|------|-------------|
| `preprocess.py` | EEG preprocessing functions |
| `filter.py` | Signal filtering utilities |
| `learning_psd.py` | EEG feature extraction experiments |
| `loocv.py` | Leave-One-Subject-Out training and evaluation pipeline |
| `shap_imp.py` | Calculates SHAP feature importance across participants |
| `overall_shap_importance.csv` | Aggregated SHAP feature ranking |
| `imp_features_run.py` | Retrains classifiers using selected SHAP features |
| `testing_features.py` | Experiments with different feature selection methods |
| `single_participant_acc.py` | Single participant evaluation |
| `all_participants_acc.py` | Evaluation across participants |
| `test.py` | Utility/testing script |
| `test_load.py` | EEG data loading test |

---

## Machine Learning Models

The following classifiers are evaluated:

- Linear Discriminant Analysis (LDA)
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

Performance is evaluated using:

- Area Under the Receiver Operating Characteristic Curve (AUC-ROC)

---

## Feature Extraction

The extracted EEG features include frequency-domain and time-domain features.

### Frequency-domain Features

- Power Spectral Density (PSD)

### Time-domain Features

- Line Length
- Skewness
- Kurtosis
- Approximate Entropy
- Sample Entropy
- Spectral Entropy
- SVD Entropy
- Hjorth Mobility
- Hjorth Complexity

---

## Feature Selection Using SHAP

Feature selection is performed using SHAP (SHapley Additive exPlanations).

The process is:

1. Train models using the extracted EEG features.
2. Calculate SHAP feature importance values.
3. Aggregate feature importance across participants.
4. Rank features based on their average SHAP importance.
5. Retrain models using different numbers of top-ranked features.

Examples of feature subsets tested include:

- Top 5 features
- Top 10 features
- Top 15 features
- Top 20 features
- Other SHAP-ranked feature subsets

---

## Dataset

The EEG dataset is not included in this repository due to privacy restrictions.

The scripts expect EEG recordings to be placed inside:
