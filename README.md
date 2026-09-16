# Learning to Predict Behaviour from EEG Dynamics

## MSc Dissertation Project

This repository contains the source code developed as part of my MSc dissertation at the University of Sheffield.

The aim of this project is to investigate whether behavioural responses can be predicted from EEG activity recorded before the behavioural response using machine learning. The analysis pipeline involves EEG signal processing, feature extraction, feature selection using SHAP, classification, cross-validation, cross-participant evaluation, and exploratory clustering.

---

## Project Overview

The workflow of the project is:

1. Loading preprocessed EEG recordings.
2. Filtering EEG signals between 1-40 Hz.
3. Selecting relevant frontal EEG channels.
4. Creating epochs from -0.9 to 0 seconds before the behavioural response.
5. Extracting frequency-domain and signal-derived features.
6. Standardising the extracted features.
7. Performing SHAP-based feature selection.
8. Training multiple classification models.
9. Evaluating within-participant classification performance.
10. Conducting Leave-One-Subject-Out (LOSO) evaluation.
11. Analysing SHAP feature importance.
12. Performing exploratory clustering analysis.

The primary objective of the project is to investigate whether pre-response EEG activity contains information that can be used to predict the subsequent behavioural response.

---

## EEG Preprocessing

The EEG data are processed using the MNE-Python library.

The EEG recordings used by the final analysis are preprocessed EEG recordings stored in EEGLAB format.

The processing steps performed by the analysis scripts include:

- Loading the cleaned EEG recordings.
- Filtering EEG data between 1-40 Hz.
- Selecting frontal EEG channels.
- Defining epochs based on behavioural events.
- Extracting the EEG segment from -0.9 to 0 seconds before the response.

The EEG channels used are:

- Fz
- FCz
- F2
- F3
- F4
- F6

The final analysis contains data from 11 participants and 800 usable epochs.

---

## Repository Structure

| File / Directory | Description |
|------|-------------|
| `one_person_loop.py` | Within-participant classification using 5-fold stratified cross-validation |
| `looc_main.py` | Leave-One-Subject-Out (LOSO) classification and evaluation |
| `main_shap_analysis.py` | Participant-level XGBoost SHAP analysis |
| `shap_imp.py` | Random Forest SHAP analysis using the LOSO framework |
| `clustering.py` | Participant-level K-Means clustering and PCA visualisation |
| `clustering_whole.py` | Combined K-Means clustering analysis across all participants |
| `data/` | Directory containing the preprocessed EEG `.set` and `.fdt` files |
| `shap_analysis/` | Directory containing SHAP analysis outputs |
| `clustering_results/` | Directory containing participant-level clustering outputs |
| `combined_clustering_results/` | Directory containing combined clustering outputs |
| `model_auc_results.csv` | LOSO model ROC-AUC results |
| `roc_curve.png` | ROC curve generated from the LOSO evaluation |
| `overall_shap_importance.csv` | Overall SHAP feature-importance results |
| `FEATURE_FREQUENCY_TOP_20.csv` | Frequency of features appearing among the most important SHAP features |

---

## Machine Learning Models

The following classifiers are evaluated:

- Linear Discriminant Analysis (LDA)
- Linear Support Vector Machine (SVM)
- Logistic Regression
- Random Forest
- XGBoost
- Soft Voting Classifier
- Dummy Classifier

Performance is evaluated using:

- Area Under the Receiver Operating Characteristic Curve (ROC-AUC)
- Classification accuracy

The Dummy Classifier is included as a baseline for comparison.

---

## Feature Extraction

The extracted EEG features consist of frequency-domain and signal-derived features.

### Frequency-domain Features

Welch Power Spectral Density (PSD) is calculated for the selected EEG channels.

The PSD representation contains:

- 6 EEG channels
- 36 frequency points per channel
- 216 PSD features in total

### Signal-derived Features

The following signal-derived features are extracted:

- Line Length
- Kurtosis
- Skewness
- Hjorth Mobility
- Hjorth Complexity
- Zero Crossings
- Spectral Entropy
- SVD Entropy
- Approximate Entropy
- Sample Entropy

These features provide:

- 6 EEG channels
- 10 features per channel
- 60 signal-derived features in total

Therefore, the complete feature representation contains:

216 PSD features + 60 signal-derived features = 276 features.

---

## Feature Scaling

The extracted features are standardised using `StandardScaler`.

For the within-participant classification pipeline, scaling is performed within each cross-validation fold.

The scaler is fitted using the training portion of the fold and is then used to transform the corresponding validation portion.

This prevents information from the held-out validation fold from being used when fitting the scaler.

---

## Feature Selection Using SHAP

Feature selection is performed using SHAP (SHapley Additive exPlanations).

For the main classification pipeline, an XGBoost classifier is used to estimate feature importance within each training fold.

The feature-selection process involves:

1. Training an XGBoost model using the training data.
2. Calculating SHAP values.
3. Calculating the mean absolute SHAP value for each feature.
4. Ranking features according to their SHAP importance.
5. Selecting the top 60 features.
6. Training the classification models using the selected features.

SHAP-based feature selection is performed using the training data within each cross-validation fold.

This prevents the held-out validation fold from being used for feature selection.

SHAP values are also analysed separately to investigate the contribution of individual EEG features to model predictions.

SHAP values represent model feature contributions and are not interpreted as evidence of causal relationships.

---

## Within-Participant Evaluation

Each participant is evaluated separately using stratified 5-fold cross-validation.

The cross-validation configuration is:

    5 folds
    Shuffle = True
    Random State = 42

For each participant:

1. EEG features are extracted.
2. The participant's data are divided into five stratified folds.
3. The feature scaler is fitted using the training fold.
4. The training data are used for SHAP-based feature selection.
5. The selected features are used to train the classification models.
6. Predictions are generated for the held-out fold.
7. Predictions from all five folds are combined.
8. Accuracy and ROC-AUC are calculated.

The resulting participant-level metrics are then used to calculate the overall model performance across participants.

---

## Leave-One-Subject-Out Evaluation

Leave-One-Subject-Out (LOSO) evaluation is used to investigate cross-participant generalisation.

For each iteration:

    1 participant = test set
    All remaining participants = training set

The process is repeated until every participant has been used as the held-out participant.

The LOSO analysis produces ROC-AUC results for each model and held-out participant.

This experiment evaluates whether patterns learned from other participants can be applied to an unseen participant.

The script is named `looc_main.py`, but the implemented evaluation is Leave-One-Subject-Out rather than Leave-One-Observation-Out.

---

## SHAP Analysis

Separate SHAP analyses are performed to investigate feature contributions to model predictions.

The analysis includes:

- Mean absolute SHAP importance
- Participant-level feature importance
- Overall feature importance
- Top-ranked features
- Lowest-ranked features
- SHAP summary plots
- SHAP beeswarm plots
- SHAP waterfall plots
- Feature-importance plots
- Feature-frequency analysis

The main participant-level SHAP analysis uses XGBoost.

A separate LOSO SHAP analysis uses Random Forest.

The SHAP analysis produces feature-level outputs that can be used to investigate which EEG features contribute most strongly to the model predictions.

---

## Exploratory Clustering

K-Means clustering is used as an exploratory analysis of the extracted EEG feature representations.

The following numbers of clusters are evaluated:

- K = 2
- K = 3
- K = 4
- K = 5

Silhouette scores are calculated for the different cluster configurations.

Principal Component Analysis (PCA) is used to project the feature representations into two dimensions for visualisation.

### Participant-Level Clustering

The `clustering.py` script performs clustering separately for each participant.

For each participant:

1. EEG features are extracted.
2. Features are standardised.
3. K-Means clustering is performed for K = 2 to K = 5.
4. Silhouette scores are calculated.
5. The clustering configuration is selected based on the silhouette score.
6. PCA is used to produce a two-dimensional visualisation.
7. Cluster assignments and summary results are saved.

The results are stored in:

    clustering_results/

A summary CSV file is also generated.

### Combined Clustering

The `clustering_whole.py` script combines the feature representations from all participants before performing K-Means clustering.

The analysis evaluates:

- K = 2
- K = 3
- K = 4
- K = 5

PCA is then used to visualise the resulting clusters.

The results are stored in:

    combined_clustering_results/

Clustering is treated as an exploratory analysis and is not part of the primary behavioural prediction evaluation.

---

## Dataset

The EEG data are not included in this repository because of data privacy and availability restrictions.

The analysis scripts expect the EEG files to be stored under:

    data/

The EEG recordings are stored in EEGLAB format and require both the `.set` files and their corresponding `.fdt` files.

Example:

    data/
    ├── c01_cleaned.set
    ├── c01_cleaned.fdt
    ├── c02_cleaned.set
    ├── c02_cleaned.fdt
    ├── c03_cleaned.set
    ├── c03_cleaned.fdt
    ├── c04_cleaned.set
    ├── c04_cleaned.fdt
    ├── c05_cleaned.set
    ├── c05_cleaned.fdt
    ├── c06_cleaned.set
    ├── c06_cleaned.fdt
    ├── c07_cleaned.set
    ├── c07_cleaned.fdt
    ├── c08_cleaned.set
    ├── c08_cleaned.fdt
    ├── c09_cleaned.set
    ├── c09_cleaned.fdt
    ├── c10_cleaned.set
    ├── c10_cleaned.fdt
    ├── c11_cleaned.set
    └── c11_cleaned.fdt

The participants included in the analysis are:

    c01
    c02
    c03
    c04
    c05
    c06
    c07
    c08
    c09
    c10
    c11

There are 800 usable epochs across the 11 participants.

---

## Running the Project

### 1. Install Requirements

Create a Python environment and install the required dependencies:

    pip install -r requirements.txt

---

### 2. Run Within-Participant Classification

Run:

    python one_person_loop.py

This performs classification separately for each participant using 5-fold stratified cross-validation.

The models evaluated include:

- LDA
- SVM
- Logistic Regression
- Random Forest
- XGBoost
- Voting Classifier
- Dummy Classifier

The analysis produces participant-level accuracy and ROC-AUC results.

---

### 3. Run LOSO Classification

Run:

    python looc_main.py

This performs Leave-One-Subject-Out evaluation.

Although the script name contains `looc`, the implemented evaluation is LOSO, where an entire participant is held out during each iteration.

The analysis generates:

    model_auc_results.csv
    roc_curve.png

---

### 4. Run Participant-Level SHAP Analysis

Run:

    python main_shap_analysis.py

This performs XGBoost SHAP analysis for each participant.

The analysis generates:

- SHAP feature values
- Feature-importance results
- Top feature rankings
- SHAP summary plots
- SHAP beeswarm plots
- SHAP waterfall plots
- Feature-frequency analysis

The outputs are stored in:

    shap_analysis/

---

### 5. Run Random Forest SHAP Analysis

Run:

    python shap_imp.py

This performs Random Forest SHAP analysis within the LOSO framework.

For each held-out participant, the Random Forest model is trained using the remaining participants and SHAP values are calculated for the held-out data.

The analysis produces overall feature-importance results, including:

    overall_shap_importance.csv

---

### 6. Run Participant-Level Clustering

Run:

    python clustering.py

This performs K-Means clustering separately for each participant.

The clustering results are stored in:

    clustering_results/

---

### 7. Run Combined Clustering

Run:

    python clustering_whole.py

This performs K-Means clustering using the combined feature data from all participants.

The results are stored in:

    combined_clustering_results/

---

## Output

The analysis generates CSV files and visualisations depending on the script being executed.

Examples of generated outputs include:

- `model_auc_results.csv`
- `roc_curve.png`
- `overall_shap_importance.csv`
- `FEATURE_FREQUENCY_TOP_20.csv`

SHAP outputs are generated inside:

    shap_analysis/

Participant-level clustering outputs are generated inside:

    clustering_results/

Combined clustering outputs are generated inside:

    combined_clustering_results/

---

## Requirements

Python 3.11+

Main dependencies:

- numpy
- pandas
- matplotlib
- scikit-learn
- mne
- mne-features
- shap
- xgboost

Install all dependencies using:

    pip install -r requirements.txt

---

## Reproducibility

A random seed of `42` is used in the main machine-learning experiments where applicable.

The primary within-participant cross-validation uses:

    StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

The main XGBoost, Random Forest, Dummy Classifier, and other applicable models use fixed random states as specified in the scripts.

---

## Notes

The repository contains some experimental and commented-out approaches that were explored during development.

These include alternative feature representations and experimental feature-selection approaches.

The primary analysis is the EEG behavioural prediction pipeline described in this README.

The main objective of the project is prediction of behavioural response from pre-response EEG activity.

The clustering analyses are exploratory and are not part of the primary behavioural prediction evaluation.

---

## Author

Adarsh Anil

MSc Cybersecurity and Artificial Intelligence

The University of Sheffield
