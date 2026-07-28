#LOOCV TESTING

import mne
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #Importing Scaling Methods
from sklearn.decomposition import PCA #Importing PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from mne_features.feature_extraction import extract_features
import shap
import pandas as pd

#Importing Models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

#Importing Evaluation Metrics
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, auc, roc_curve

#For file handling
import glob
import os

mne.set_log_level("ERROR")
files = sorted(glob.glob("data/*.set"))

def preprocess(X_train, X_test):
    
    #Scaling the values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #PCA done with 90% variance
    pca = PCA(n_components=0.90)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, X_test

def get_psd_features(epochs):

    # Computing PSD (Power Spectral Density) for all frequencies
    psds = epochs.compute_psd(method="welch", fmin=1, fmax=40)

    psd_features = psds.get_data()

    #Flattening the EEG data (ML Models need 2D data)
    psd_features = psd_features.reshape(len(psd_features), -1)

    return psds, psd_features

def get_mne_features(epochs):

    #Getting EEG signal data
    data = epochs.get_data()

    #Extracting MNE Features
    features = extract_features(
    data,
    sfreq=epochs.info["sfreq"],
    selected_funcs=[
        "kurtosis",
        "skewness",
        "hjorth_mobility",
        "hjorth_complexity",
        "spect_entropy",
        "svd_entropy",
        "samp_entropy"
        ]
    )

    return features

def get_features_labels(file_path):
    
    #Reading the data from EEG data using MNE
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    
    #Filtering between 1-40 hz to remove Low frequencies and Muscle Noises
    raw = raw.copy().filter(1, 40)

    #Selecting only frontal EEG channels
    channels = ["Fz", "FCz", "F2", "F3", "F4", "F6"]
    raw = raw.pick(channels)

    #Creating events based on conditions present on data
    events, event_id = mne.events_from_annotations(raw)

    #Creating epochs - 1 second of data from start of condition
    epochs = mne.Epochs(
    raw,
    events,
    event_id={
        "condition 1": event_id["condition 1"],
        "condition 2": event_id["condition 2"]
    },
    tmin=-0.9,
    tmax=0,
    baseline=None,
    preload=True
    )

    #Getting PSD Features
    psds, psd_features = get_psd_features(epochs)

    #The Y Labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(epochs.events[:, -1])

    #MNE Features (Experiment 5)
    mne_features = get_mne_features(epochs)

    #Combining PSD + MNE Features (Experiment 6)
    X = np.concatenate((psd_features, mne_features), axis=1)

    #Creating names for every feature
    feature_names = []

    #PSD feature names
    freqs = psds.freqs

    for channel in channels:
        for frequency in freqs:
            feature_names.append(
                "PSD_" + channel + "_" + str(round(frequency, 1)) + "Hz"
            )

   #Names of the MNE features
    mne_feature_names = [
    "kurtosis",
    "skewness",
    "hjorth_mobility",
    "hjorth_complexity",
    "spect_entropy",
    "svd_entropy",
    "samp_entropy"
    ]     

    #Adding the feature name for every channel
    for feature in mne_feature_names:
        for channel in channels:
            feature_names.append(channel + "_" + feature)

    print("Feature Shape:", X.shape)

    return X, y, feature_names

def select_top_features(X_train, y_train, feature_names, n_features=10):

    #Temporary model for finding important features
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    #Train only on training data
    model.fit(X_train, y_train)

    #Getting feature importance
    importance = model.feature_importances_

    #Creating dataframe of features and importance
    shap_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    #Sorting features by importance
    shap_importance = shap_importance.sort_values(
        by="Importance",
        ascending=False
    )

    #Selecting top features
    selected_features = shap_importance.head(n_features)["Feature"].tolist()

    #Getting column indexes
    selected_indices = [
        feature_names.index(feature)
        for feature in selected_features
    ]

    print("\nSelected Features:")
    print(selected_features)

    print("\nNumber of Selected Features:", len(selected_indices))

    return selected_indices

#FUNCTION TO RUN THE LDA MODEL----------------------
def run_lda(X_train, X_test, y_train, y_test):

    #Calling  Preprocessing Function
    X_train, X_test = preprocess(X_train, X_test)

    #MODEL 1: LINEAR DISCRIMINANT ANALYSIS (LDA)
    model = LinearDiscriminantAnalysis()
    
    #Training the Model
    model.fit(X_train, y_train)

    #Predicting the Y Labels on the Test Dataset
    y_prob = model.predict_proba(X_test)[:,1]

    #Accuracy
    #return accuracy_score(y_test, y_pred)

    #AUC-ROC Score
    auc_score = roc_auc_score(y_test, y_prob)
    return auc_score, y_test, y_prob

#FUNCTION TO RUN THE SVM MODEL----------------------
def run_svm(X_train, X_test, y_train, y_test):

    #Calling Preprocessing function
    X_train, X_test = preprocess(X_train, X_test)

    #MODEL 2: SUPPORT VECTOR MACHINE (SVM) - new way because of changed versions
    svm = SVC(kernel="linear", random_state=42)
    
    #Calibration to get probabilities
    model = CalibratedClassifierCV(
        svm,
        method="sigmoid",
        cv=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42)
    )
    
    #Training the Model
    model.fit(X_train, y_train)

    #Predicting the Y Labels on Test Dataset
    y_prob = model.predict_proba(X_test)[:,1]

    #Accuracy
    #return accuracy_score(y_test, y_pred)

    #Return AUC-ROC Score
    auc_score = roc_auc_score(y_test, y_prob)
    return auc_score, y_test, y_prob

#FUNCTION TO RUN THE RANDOM FOREST MODEL----------------------
def run_random_forest(X_train, X_test, y_train, y_test):

    #Calling Preprocessing Function
    X_train, X_test = preprocess(X_train, X_test)

    #MODEL 3: RANDOM FOREST
    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42
    )

    #Training the Model
    model.fit(X_train, y_train)

    #Predicting the Y Labels on Test Dataset
    y_prob = model.predict_proba(X_test)[:,1]

    #Accuracy
    #return accuracy_score(y_test, y_pred)

    #Return AUC-ROC Score
    auc_score = roc_auc_score(y_test, y_prob)
    return auc_score, y_test, y_prob

#FUNCTION TO RUN THE XGBOOST MODEL----------------------
def run_xgboost(X_train, X_test, y_train, y_test):

    #Calling Preprocessing Function
    X_train, X_test = preprocess(X_train, X_test)

    #MODEL 4: XGBOOST
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )

    #Training the Model
    model.fit(X_train, y_train)

    #Predicting the Y Labels
    y_prob = model.predict_proba(X_test)[:,1]

    #Accuracy
    #return accuracy_score(y_test, y_pred)

    #Return AUC-ROC Score
    auc_score = roc_auc_score(y_test, y_prob)
    return auc_score, y_test, y_prob

#FUNCTION TO RUN DUMMY CLASSIFIER ----------------------
def run_dummy(X_train, X_test, y_train, y_test):

    #Dummy classifier uses the most frequent class
    model = DummyClassifier(strategy="stratified", random_state=42)

    #Training
    model.fit(X_train, y_train)

    #Prediction probabilities
    y_prob = model.predict_proba(X_test)[:,1]

    #AUC score
    auc_score = roc_auc_score(y_test, y_prob)

    return auc_score, y_test, y_prob

#Main Code - Body
start_time = time.time()

lda_scores = []
svm_scores = []
rf_scores = []
xgb_scores = []
dummy_scores = []

results_table = []

roc_labels = []
lda_probs = []
svm_probs = []
rf_probs = []
xgb_probs = []
dummy_probs = []

#FOR storing values of all X and Y
X_all = []
y_all = []

##Adding X and Y to the variables
for file in files:
    X, y, feature_names = get_features_labels(file)
    X_all.append(X)
    y_all.append(y)

X_all = np.vstack(X_all)
y_all = np.concatenate(y_all)

#Starting Stratified K-Fold
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for fold, (train_index, test_index) in enumerate(skf.split(X_all, y_all), start=1):

    print("\nFold:", fold)

    #Split data
    X_train = X_all[train_index]
    X_test = X_all[test_index]

    y_train = y_all[train_index]
    y_test = y_all[test_index]

    #Load top SHAP features
    selected_indices = select_top_features(X_train, y_train, feature_names, n_features=10)

    #Select the same features from train and test
    X_train = X_train[:, selected_indices]
    X_test = X_test[:, selected_indices]

    #Running LDA
    lda_acc, lda_y, lda_prob = run_lda(X_train, X_test, y_train, y_test)

    #Running SVM
    svm_acc, svm_y, svm_prob = run_svm(X_train, X_test, y_train, y_test)

    #Running Random Forest
    rf_acc, rf_y, rf_prob = run_random_forest(X_train, X_test, y_train, y_test)

    #Running XGBoost
    xgb_acc, xgb_y, xgb_prob = run_xgboost(X_train, X_test, y_train, y_test)

    #Running Dummy
    dummy_acc, dummy_y, dummy_prob = run_dummy(X_train, X_test, y_train, y_test)

    roc_labels.extend(y_test)

    lda_probs.extend(lda_prob)
    svm_probs.extend(svm_prob)
    rf_probs.extend(rf_prob)
    xgb_probs.extend(xgb_prob)
    dummy_probs.extend(dummy_prob)

    print("\nLDA AUC :", lda_acc)
    print("SVM AUC :", svm_acc)
    print("Random Forest AUC :", rf_acc)
    print("XGBoost AUC :", xgb_acc)
    print("Dummy AUC :", dummy_acc)

    results_table.append({
        "Fold": fold,
        "LDA_AUC": lda_acc,
        "SVM_AUC": svm_acc,
        "RandomForest_AUC": rf_acc,
        "XGBoost_AUC": xgb_acc,
        "Dummy_AUC": dummy_acc
    })

    lda_scores.append(lda_acc)
    svm_scores.append(svm_acc)
    rf_scores.append(rf_acc)
    xgb_scores.append(xgb_acc)
    dummy_scores.append(dummy_acc)

#Final Results
print("\n Final Results")
print("\nAverage LDA AUC :", np.mean(lda_scores))
print("\nAverage SVM AUC :", np.mean(svm_scores))
print("\nAverage Random Forest AUC :", np.mean(rf_scores))
print("\nAverage XGBoost AUC :", np.mean(xgb_scores))
print("\nAverage Dummy AUC :", np.mean(dummy_scores))

#Savign results in a CSV Table
results_df = pd.DataFrame(results_table)
average_row = {
    "Fold": "Average",
    "LDA_AUC": np.mean(lda_scores),
    "SVM_AUC": np.mean(svm_scores),
    "RandomForest_AUC": np.mean(rf_scores),
    "XGBoost_AUC": np.mean(xgb_scores),
    "Dummy_AUC": np.mean(dummy_scores)
}
results_df.loc[len(results_df)] = average_row
results_df.to_csv("model_auc_results.csv", index=False)
print("\nResults saved as model_auc_results.csv")

#ROC CURVE
models = {
    "LDA": lda_probs,
    "SVM": svm_probs,
    "Random Forest": rf_probs,
    "XGBoost": xgb_probs,
    "Dummy": dummy_probs
}

plt.figure(figsize=(8,6))

for name, probs in models.items():
    fpr, tpr, thresholds = roc_curve(roc_labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

#Random classifier line
plt.plot([0,1], [0,1], linestyle="--", label="Chance")

#Make the graph
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - 5-Fold EEG Classification")
plt.legend()
plt.grid(True)

#Save the results
plt.savefig("roc_curve_f_fold.png", dpi=300, bbox_inches="tight")
plt.show()
print("ROC curve saved")

#time to run program being printed -----------------------------------------
end_time = time.time()
runtime = end_time - start_time
print(f"\nTotal Runtime: {runtime:.2f} seconds")
print(f"Total Runtime: {runtime/60:.2f} minutes")