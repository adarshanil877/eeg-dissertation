# Learning to Predict Behaviour from EEG Dynamics

## MSc Dissertation Project

This repository holds the source code that was written as part of my MSc dissertation at the University of Sheffield.

The goal of this project is to explore the possibility of predicting behavioural patterns from EEG signals by means of machine learning. The pipeline of the project involves EEG analysis, which consists of signal pre-processing, feature extraction, feature selection via SHAP, and classification.

---

## Project Overview

The workflow of the project is:

1. Loading raw EEG recordings.
2. EEG signal preprocessing.
3. Feature extraction in frequency domain and time domain.
4. Conducting Leave-One-Subject-Out (LOSO) test.
5. Estimating SHAP feature importance for each participant.
6. Selection of top features.
7. Retraining classifiers based on selected features.
8. Classification performance comparison by AUC-ROC score.

---

## EEG Preprocessing

The data obtained through EEG is then processed through the MNE-Python library.

The pre-processing steps include:

- Filtering of EEG data between 1-40Hz.
- Selecting frontal EEG channels.
- Defining epochs on the basis of events.
- Isolating 1 second EEG data before extracting features from them.

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

Feature selection is done by using SHAP (SHapley Additive exPlanations).

This involves the following steps:

1. Train the models with the extracted EEG features.
2. Compute SHAP feature importance scores.
3. Compute the average feature importance over the participants.
4. Rank the features based on the average SHAP importance.
5. Retrain the models with varying number of selected features.

Some of the feature subsets used include:

- The top 5 features
- The top 10 features
- The top 15 features
- The top 20 features
- Others

---

## Dataset

The EEG dataset is not included in this repository due to privacy restrictions.

The scripts expect EEG recordings to be placed inside:
