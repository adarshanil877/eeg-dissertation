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
| `preprocess.py` | Functions for EEG pre-processing |
| `filter.py` | Signal filtering tools |
| `learning_psd.py` | EEG feature extraction experiments |
| `loocv.py` | Leave-one-out training & evaluation framework  |
| `shap_imp.py` | Calculate SHAP feature importances for all subjects |
| `overall_shap_importance.csv` | SHAP feature importances summary |
| `imp_features_run.py` | Classifiers retraining on SHAP selected features |
| `testing_features.py` | Experiment with various feature selection techniques |
| `single_participant_acc.py` | Single subject evaluation |
| `all_participants_acc.py` |  Participants’ evaluation |
| `test.py` | General utility testing file |
| `test_load.py` | Test EEG data loading |

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

The EEG data will not be available within this repository because of privacy reasons.

EEG data are expected to be stored under the following path:

```
data/
```

The files should be named according to the participant naming convention used within the project.

---

## Running the Project

### 1. Install Requirements

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

---

### 2. Run Baseline Classification Pipeline

Run the original LOOCV:

```bash
python loocv.py
```

---

### 3. Generate SHAP Feature Importance

Run:

```bash
python shap_imp.py
```

This generates:

```
overall_shap_importance.csv
```

which contains the ranked feature importance values.

---

### 4. Run Classification with Selected Features

Run:

```bash
python imp_features_run.py
```

This trains and evaluates models using SHAP-selected features.

---

## Current Results

The first LOOCV tests resulted in an AUC-ROC of about 0.60.

The use of SHAP feature selection resulted in a decrease of input variables while achieving similar performance.

---

## Requirements

Python 3.11+

Main dependencies:

- numpy
- pandas
- scipy
- scikit-learn
- mne
- mne-features
- shap
- xgboost
- matplotlib

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Author

Adarsh Anil

MSc Cybersecurity and Artificial Intelligence

The University of Sheffield
