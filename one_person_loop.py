#ALL PARTICIPANTS INDIVIDUAL TESTING 
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

#IMPORTING MODELS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

#IMPORTING EVALUATION METRICS
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, auc, roc_curve

#IMPORTS FOR FILE HANDLING
import glob
import os

mne.set_log_level("ERROR")
files = sorted(glob.glob("data/*.set"))

#ONE PARTICIPANT SELECTED FOR EXPERIMENT
#file = files[0]
#print("PARTICIPANT:", os.path.basename(file))

def preprocess(X_train, X_test):
    
    #Scaling the values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #PCA done with 90% variance
    #pca = PCA(n_components=0.90)
    #X_train = pca.fit_transform(X_train)
    #X_test = pca.transform(X_test)

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
        "line_length",
        "kurtosis",
        "skewness",
        "hjorth_mobility",
        "hjorth_complexity",
        "zero_crossings",
        "spect_entropy",
        "svd_entropy",
        "app_entropy",
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

    #MNE Features (Experiment 5)
    mne_features = get_mne_features(epochs)

    #Combining PSD + MNE Features (Experiment 6)
    X = np.concatenate((psd_features, mne_features), axis=1)

    #The Y Labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(epochs.events[:, -1])

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
        "line_length",
        "kurtosis",
        "skewness",
        "hjorth_mobility",
        "hjorth_complexity",
        "zero_crossings",
        "spect_entropy",
        "svd_entropy",
        "app_entropy",
        "samp_entropy"
    ]

    for feature in mne_feature_names:
        for channel in channels:
            feature_names.append(channel + "_" + feature)


    print("FEATURE SHAPE :", X.shape)

    return X, y, feature_names

def get_shap_top_features(X_train_scaled, y_train, feature_names, n_features=10):

    #XGBoost used to get SHAP Values
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train_scaled, y_train)

    #Computing SHAP values on the training fold only
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_scaled)

    #Averaging absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    #Ranking features by importance, highest first
    ranking = np.argsort(mean_abs_shap)[::-1]
    selected_indices = ranking[:n_features].tolist()
    selected_features = [feature_names[i] for i in selected_indices]

    #print("\nSELECTED FEATURES :")
    #for feature in selected_features:
         #print(feature)

    return selected_indices, selected_features

#FUNCTION TO RUN THE LDA MODEL----------------------
def run_lda(X_train, X_test, y_train, y_test):

    #MODEL 1: LINEAR DISCRIMINANT ANALYSIS (LDA)
    model = LinearDiscriminantAnalysis()
    
    #Training the Model
    model.fit(X_train, y_train)

    #Predicting the Y Labels on the Test Dataset
    y_prob = model.predict_proba(X_test)[:,1]

    #AUC-ROC  and Accuracy Score
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)

    return accuracy, auc_score, y_test, y_prob

#FUNCTION TO RUN THE SVM MODEL----------------------
def run_svm(X_train, X_test, y_train, y_test):

    #MODEL 2: SUPPORT VECTOR MACHINE (SVM) - new way because of changed versions
    svm = SVC(kernel="linear", random_state=42)
    
    #Calibration to get probabilities
    model = CalibratedClassifierCV(
    svm,
    method="sigmoid",
    cv=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
        )
    )
    
    #Training the Model
    model.fit(X_train, y_train)

    #Predicting the Y Labels on Test Dataset
    y_prob = model.predict_proba(X_test)[:,1]

    #AUC-ROC  and Accuracy Score
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)

    return accuracy, auc_score, y_test, y_prob

#FUNCTION TO RUN LOGISTIC REGRESSION ----------------------
def run_logistic_regression(X_train, X_test, y_train, y_test):

    #MODEL: LOGISTIC REGRESSION
    model = LogisticRegression(
        C=0.1,
        max_iter=1000,
        random_state=42
    )

    #Training
    model.fit(X_train, y_train)

    #Probability prediction
    y_prob = model.predict_proba(X_test)[:,1]

    #AUC-ROC  and Accuracy Score
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)

    return accuracy, auc_score, y_test, y_prob

#FUNCTION TO RUN THE RANDOM FOREST MODEL----------------------
def run_random_forest(X_train, X_test, y_train, y_test):

    #MODEL 3: RANDOM FOREST
    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42
    )

    #Training the Model
    model.fit(X_train, y_train)

    #Predicting the Y Labels on Test Dataset
    y_prob = model.predict_proba(X_test)[:,1]

    #AUC-ROC  and Accuracy Score
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)

    return accuracy, auc_score, y_test, y_prob

#FUNCTION TO RUN THE XGBOOST MODEL----------------------
def run_xgboost(X_train, X_test, y_train, y_test):

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

    #AUC-ROC  and Accuracy Score
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)

    return accuracy, auc_score, y_test, y_prob

#FUNCTION TO RUN DUMMY CLASSIFIER ----------------------
def run_dummy(X_train, X_test, y_train, y_test):

    #Dummy classifier uses the most frequent class
    model = DummyClassifier(strategy="stratified", random_state=42)

    #Training
    model.fit(X_train, y_train)

    #Prediction probabilities
    y_prob = model.predict_proba(X_test)[:,1]

    #AUC-ROC  and Accuracy Score
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)

    return accuracy, auc_score, y_test, y_prob

#MAIN BODY
start_time = time.time()

print("\nRUNNING ALL PARTICIPANTS")

models = {
    "LDA": run_lda,
    "SVM": run_svm,
    "LOGISTIC": run_logistic_regression,
    "RANDOM FOREST": run_random_forest,
    "XGBOOST": run_xgboost,
    "DUMMY": run_dummy
}

results = {
    "LDA": [],
    "SVM": [],
    "LOGISTIC": [],
    "RANDOM FOREST": [],
    "XGBOOST": [],
    "DUMMY": []
}

for file in files:

    #STORAGE
    all_y_true = []
    all_lda_prob = []
    all_svm_prob = []
    all_rf_prob = []
    all_xgb_prob = []
    all_log_prob = []
    all_dummy_prob = []
    fold_auc_results = []

    participant = os.path.basename(file).replace("_cleaned.set", "")

    print("\nPARTICIPANT : ", participant)

    #EXTRACTING FEATURES FROM ONE PARTICIPANT
    X,y, feature_names = get_features_labels(file)
    #print("\nTOTAL SAMPLES: ", X.shape[0])
    #print("TOTAL FEATURES: ", X.shape[1])

    #K-FOLD CROSS VALIDATIONS
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nSTARTING 5-FOLD CROSS-VALIDATION FOR PARTICIPANT : ", participant)

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X,y), start=1):

        #print("\nFOLD : ", fold)

        #DATA SPLIT INTO TRAINING AND TESTING DATA
        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        #print("TRAINING SAMPLES : ", len(train_idx))
        #print("TESTING SAMPLES : ", len(test_idx))

        scaler=StandardScaler()
        
        #SCALING THE TRAINING DATA
        X_train_scaled = scaler.fit_transform(X_train)
        
        #SCALING THE VALIDATION DATA SEPERATELY
        X_test_scaled = scaler.transform(X_test)

        #SHAP FEATURE SELECTION
        selected_indices, selected_features = get_shap_top_features(X_train_scaled, y_train, feature_names, n_features=60)

        #SELECTING THE REQUIRED SHAP FEATURES ONLY:
        X_train_selected = X_train_scaled[:, selected_indices]
        X_test_selected = X_test_scaled[:, selected_indices]

        #print("\nTRAINING SHAPE AFTER SHAP : ", X_train_selected.shape)
        #print("TESTING SHAPE AFTER SHAP : ", X_test_selected.shape)

        #Running LDA
        lda_accuracy, lda_auc, lda_y, lda_prob = run_lda(X_train_selected, X_test_selected, y_train, y_test)

        #Running SVM
        svm_accuracy, svm_auc, svm_y, svm_prob = run_svm(X_train_selected, X_test_selected, y_train, y_test)

        #Running Random Forest
        rf_accuracy, rf_auc, rf_y, rf_prob = run_random_forest(X_train_selected, X_test_selected, y_train, y_test)

        #Running XGBoost
        xgb_accuracy, xgb_auc, xgb_y, xgb_prob = run_xgboost(X_train_selected, X_test_selected, y_train, y_test)

        #Running Logistic Regression
        log_accuracy, log_auc, log_y, log_prob = run_logistic_regression(X_train_selected, X_test_selected, y_train, y_test)

        #Running Dummy Classifier
        dummy_accuracy, dummy_auc, dummy_y, dummy_prob = run_dummy(X_train_selected, X_test_selected, y_train, y_test)
        
        #SAVING PREDICTIONS OF THIS FOLD TO COMPUTE OVERALL FOLD RESULTS
        all_y_true.extend(y_test)
        all_lda_prob.extend(lda_prob)
        all_svm_prob.extend(svm_prob)
        all_log_prob.extend(log_prob)
        all_rf_prob.extend(rf_prob)
        all_xgb_prob.extend(xgb_prob)
        all_dummy_prob.extend(dummy_prob)

    #OVERALL APRTICIPANT AUC
    participant_lda = roc_auc_score(all_y_true, all_lda_prob)
    participant_svm = roc_auc_score(all_y_true, all_svm_prob)
    participant_log = roc_auc_score(all_y_true, all_log_prob)
    participant_rf = roc_auc_score(all_y_true, all_rf_prob)
    participant_xgb = roc_auc_score(all_y_true, all_xgb_prob)
    participant_dummy = roc_auc_score(all_y_true, all_dummy_prob)

    #SAVING RESULTS
    results["LDA"].append(participant_lda)
    results["SVM"].append(participant_svm)
    results["LOGISTIC"].append(participant_log)
    results["RANDOM FOREST"].append(participant_rf)
    results["XGBOOST"].append(participant_xgb)
    results["DUMMY"].append(participant_dummy)

    #PRINTING RESULTS FOR FIRST PARTICIPANT
    print("LDA AUC :", round(participant_lda, 4))
    print("SVM AUC :", round(participant_svm, 4))
    print("LOGISTIC AUC :", round(participant_log, 4))
    print("RANDOM FOREST AUC :", round(participant_rf, 4))
    print("XGBOOST AUC :", round(participant_xgb, 4))
    print("DUMMY AUC :", round(participant_dummy, 4))

#OVERALL AUC RESULTS
print("\nFINAL PARTICIPANT RESULTS")
for i, file in enumerate(files):
    participant = os.path.basename(file).replace("_cleaned.set", "")
    print(participant,
        "LDA : ", round(results["LDA"][i], 4),
        "SVM : ", round(results["SVM"][i], 4),
        "LOGISTIC : ", round(results["LOGISTIC"][i], 4),
        "RF : ", round(results["RANDOM FOREST"][i], 4),
        "XGB : ", round(results["XGBOOST"][i], 4),
        "DUMMY : ", round(results["DUMMY"][i], 4)
    )

#AVERAGE AUC FOR EACH MODEL
print("\nAVERAGE AUC")
for model_name in results:
    average_auc = np.mean(results[model_name])
    print(model_name, ":", round(average_auc, 4))

#TIME TO RUN PROGRAM
end_time = time.time()
runtime = end_time - start_time
print(f"\nTOTAL RUNTIME (SECS): {runtime:.2f} seconds")
print(f"TOTAL RUNTIME (MINS): {runtime/60:.2f} minutes")