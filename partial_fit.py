#PARTIAL FIT LOOCV

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
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier

#Importing Evaluation Metrics
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, auc, roc_curve

#For file handling
import glob
import os

mne.set_log_level("ERROR")
files = sorted(glob.glob("data/*.set"))

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

    #The Y Labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(epochs.events[:, -1])
    
    #MNE Features (Experiment 5)
    mne_features = get_mne_features(epochs)

    #X = mne_features
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

    #Adding the feature name for every channel
    for feature in mne_feature_names:
        for channel in channels:
            feature_names.append(channel + "_" + feature)

    print("Feature Shape:", X.shape)

    return X, y, feature_names

def get_shap_top_features(X_train, y_train, feature_names, n_features=10):

    #XGBoost used to get SHAP Values
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    #Computing SHAP values on the training fold only
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    #Averaging absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    #Ranking features by importance, highest first
    ranking = np.argsort(mean_abs_shap)[::-1]
    selected_indices = ranking[:n_features].tolist()
    selected_features = [feature_names[i] for i in selected_indices]

    print("\nSelected Features:")
    print(selected_features)
    print("\nNumber of Selected Features:", len(selected_indices))

    return selected_indices, selected_features

#MAIN
start_time = time.time()
print("\nLOOCV + PARTIAL FIT")

#LOADING ALL PARTICIPANTS

#Dictionary to store all participants
participants = {}

#Reading every participant
for file in files:

    X, y, feature_names = get_features_labels(file)

    #Using participant names from filenames
    participant_name = os.path.basename(file).replace("_cleaned.set", "")

    #Adding the features to dictionary
    participants[participant_name] = {
        "X": X,
        "y": y,
        "feature_names": feature_names
    }

#STORAGE
before_scores = []
after_scores = []
results_table = []
roc_labels = []
before_probs = []
after_probs = []

#STARTING LOOCV
for test_subject in participants:
    print("\nTHE PARTICIPANT USED FOR FINE-TUNING : ", test_subject)

    #TRAINING PARTICIPANTS:
    train_subjects = []
    
    #Making the Training subjects
    for participant in participants:
        if participant != test_subject:
            train_subjects.append(participant)
    
    #Showing the Training Participants
    print("Training Participants:", train_subjects)

    X_train_full=[]
    y_train_full=[]

    for participant in train_subjects:
            X_train_full.append(participants[participant]["X"])
            y_train_full.append(participants[participant]["y"])

    X_train_full = np.vstack(X_train_full)
    y_train_full = np.concatenate(y_train_full)

    #SCALING ONLY TRAINING PARTICIPANTS
    scaler = StandardScaler()
    X_train_scaled = (scaler.fit_transform(X_train_full))  

    #SHAP FEATURE SELECTION
    #Selecting top features using SHAP
    selected_indices, selected_features = get_shap_top_features(
        X_train_scaled,
        y_train_full,
        feature_names,
        n_features=10
    )

    #Selecting Training Features
    X_train_selected = (X_train_scaled[:, selected_indices])

    #THE FINE TUNING PARTICIPANT
    X_test = participants[test_subject]["X"]
    y_test = participants[test_subject]["y"]

    #Scaling the Test Subject
    X_test_scaled = scaler.transform(X_test)

    #SHAP for test subjects
    X_test_selected = X_test_scaled[:, selected_indices]

    #SPLITING TEST PARTICIPANT
    n=len(X_test_selected)

    split_point = n//2

    #THE FIRST HALF IS MOVED ASIDE FOR ADAPTATION
    X_adapt = X_test_selected[:split_point]
    y_adapt = y_test[:split_point]

    #SECOND HALF IS FOR TESTING
    X_maintest = X_test_selected[split_point:]
    y_maintest = y_test[split_point:]

    print("\nTOTAL SAMPLES IN TEST PARTICIPANT : ", n)
    print("FINE TUNING SAMPLES :", len(X_adapt))
    print("FINAL TEST SAMPLES : ", len(X_maintest))

    #MODEL
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0001,
        max_iter=1000,
        random_state=42
    )

    #TRAINING ON 10 PARTICIPANTS
    model.fit(X_train_selected, y_train_full)

    #STEP 1 : TEST BEFORE FINE TUNING - NORMAL METHOD
    before_prob = model.predict_proba(X_maintest)[:,1]
    before_auc = roc_auc_score(y_maintest,before_prob)

    #STEP 2: FINE TUNING PN FIRST HALF AND THEN TESTING
    model.partial_fit(X_adapt,y_adapt,classes=np.array([0,1]))

    #STEP 3: TEST ON SECOND HALF OF TEST SUBJECT
    after_prob = model.predict_proba(X_maintest)[:,1]
    after_auc = roc_auc_score(y_maintest,after_prob)

    #RESULTS
    improvement = after_auc-before_auc

    print("\nBEFORE FINE-TUNING AUC : ", round(before_auc, 4))
    print("AFTER FINE-TUNING AUC : ", round(after_auc, 4))
    print("IMPROVEMENT : ", round(improvement, 4))

    before_scores.append(before_auc)
    after_scores.append(after_auc)

    results_table.append({
        "Participant":test_subject,
        "Before_FineTune_AUC":before_auc,
        "After_FineTune_AUC":after_auc,
        "Improvement":improvement
    })

     # ROC data
    roc_labels.extend(y_test)
    before_probs.extend(before_prob)
    after_probs.extend(after_prob)

#FINAL RESULTS
print("FINAL RESULTS")
print("\nAVERAGE AUC BEFORE FINE TUNING : ",round(np.mean(before_scores),4))

print("AVERAGE AUC AFTER FINE TUNING : ",round(np.mean(after_scores),4))
print("\nMEAN IMPROVEMENT : ",round(np.mean(np.array(after_scores)-np.array(before_scores)),4))
print("STANDARD DEVIATION BEFORE:",round(np.std(before_scores),4))
print("STANDARD DEVIATION AFTER:", round(np.std(after_scores),4))

#RESULTS TABLE
results_df = pd.DataFrame(results_table)
results_df.to_csv("partial_fit_results.csv", index=False)
print("\nRESULTS SAVED IN partial_fit_results.csv")   

    
