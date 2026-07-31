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
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

#Importing Evaluation Metrics
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, auc, roc_curve

#For file handling
import glob
import os

mne.set_log_level("ERROR")
files = sorted(glob.glob("data/*.set"))

ID_TRANSITIONS = [
    (23,24),
    (22,21),
    (31,32),
    (32,31)
]

ED_TRANSITIONS = [
    (13,12),
    (12,14),
    (14,11),
    (11,23),
    (24,22),
    (21,31),
    (31,33)
]

EXPERIMENT = "ALL"
# EXPERIMENT = "ID"
# EXPERIMENT = "ED"

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

def get_transition_labels(events):

    #Creating a label for every event
    labels = ["NONE"] * len(events)

    #List to store only stimulus events
    stimulus_events = []

    #Looping through every event
    for i, event in enumerate(events):

        #Getting the trigger code
        trigger = event[2]

        #Ignoring correct, incorrect and blank screen triggers
        if trigger not in [1,2,10]:
            stimulus_events.append((i, trigger))

    #Initially no transition has occurred
    current_label = "NONE"

    #Checking consecutive stimulus pairs
    for k in range(len(stimulus_events)-1):

        #Current and next stimulus
        idx1, trig1 = stimulus_events[k]
        idx2, trig2 = stimulus_events[k+1]

        #Creating the transition pair
        pair = (trig1, trig2)

        #Checking if the transition is an ID shift
        if pair in ID_TRANSITIONS:
            current_label = "ID"

        #Checking if the transition is an ED shift
        elif pair in ED_TRANSITIONS:
            current_label = "ED"

        #Assigning the transition label to all events until the next stimulus
        for j in range(idx1, idx2):
            labels[j] = current_label

    #Returning the label for every event
    return labels

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

    #Getting transition labels
    transition_labels = get_transition_labels(events)

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

    epoch_labels = []

    #Getting ID, ED and NONE for each epoch
    for sample in epochs.events[:,0]:

        idx = np.where(events[:,0] == sample)[0][0]

        epoch_labels.append(
            transition_labels[idx]
        )

    #Keeping only selected transition type
    if EXPERIMENT != "ALL":
        keep = np.array(epoch_labels) == EXPERIMENT
        epochs = epochs[keep]
        epoch_labels = list(np.array(epoch_labels)[keep])

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
    model = RandomForestClassifier(
    n_estimators=50,
    random_state=42
    )
    model.fit(X_train, y_train)

    #Computing SHAP values on the training fold only
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    #Handle different SHAP output formats
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

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
        random_state=42
        )
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

#FUNCTION TO RUN LOGISTIC REGRESSION ----------------------
def run_logistic_regression(X_train, X_test, y_train, y_test):

    #Scaling
    X_train, X_test = preprocess(X_train, X_test)

    #MODEL: Logistic Regression
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    #Training
    model.fit(X_train, y_train)

    #Probability prediction
    y_prob = model.predict_proba(X_test)[:,1]

    #AUC
    auc_score = roc_auc_score(y_test, y_prob)

    return auc_score, y_test, y_prob

#FUNCTION TO RUN THE RANDOM FOREST MODEL----------------------
def run_random_forest(X_train, X_test, y_train, y_test):

    #Calling Preprocessing Function
    #X_train, X_test = preprocess(X_train, X_test)

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
    #X_train, X_test = preprocess(X_train, X_test)

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


lda_scores = []
svm_scores = []
rf_scores =[]
xgb_scores = []
log_scores = []
all_shap_values = []
dummy_scores = []
results_table = []

#For ROC Curve
roc_labels = []
lda_probs = []
svm_probs = []
rf_probs = []
xgb_probs = []
log_probs = []
dummy_probs = []

#Starting LOOCV
for test_subject in participants:

    print("\nTesting on:", test_subject)    
    train_subjects = []

    #Making the Training subjects
    for participant in participants:
        if participant != test_subject:
            train_subjects.append(participant)

    #Showing the Training Participants
    print("Training Participants:", train_subjects)

    #Combining only the Training Participants
    X_train_full = []
    y_train = []

    for participant in train_subjects:

        X_train_full.append(
            participants[participant]["X"]
        )

        y_train.append(
            participants[participant]["y"]
        )

    X_train_full = np.vstack(X_train_full)
    y_train = np.concatenate(y_train)

    #Selecting top features using SHAP
    selected_indices, selected_features = get_shap_top_features(
        X_train_full,
        y_train,
        feature_names,
        n_features=10
    )

    #Applying the selected features to train and test sets
    X_train = X_train_full[:, selected_indices]

    #Testing Participant
    X_test = participants[test_subject]["X"][:, selected_indices]
    y_test = participants[test_subject]["y"]

    #Running LDA
    lda_acc, lda_y, lda_prob = run_lda(X_train, X_test, y_train, y_test)

    #Running SVM
    svm_acc, svm_y, svm_prob = run_svm(X_train, X_test, y_train, y_test)

    #Running Random Forest
    rf_acc, rf_y, rf_prob = run_random_forest(X_train, X_test, y_train, y_test)

    #Running XGBoost
    xgb_acc, xgb_y, xgb_prob = run_xgboost(X_train, X_test, y_train, y_test)

    #Running Logistic Regression
    log_acc, log_y, log_prob = run_logistic_regression(X_train, X_test, y_train, y_test)

    #Running Dummy Classifier
    dummy_acc, dummy_y, dummy_prob = run_dummy(X_train, X_test, y_train, y_test)

    #Labels are stored
    roc_labels.extend(y_test)

    #Prediction Probabilities Stored
    lda_probs.extend(lda_prob)
    svm_probs.extend(svm_prob)
    rf_probs.extend(rf_prob)
    xgb_probs.extend(xgb_prob)
    log_probs.extend(log_prob)
    dummy_probs.extend(dummy_prob)

    #Printing AUC Scores
    print("\nLDA AUC :", lda_acc)
    print("SVM AUC :", svm_acc)
    print("Random Forest AUC :", rf_acc)
    print("XGBoost AUC :", xgb_acc)
    print("Logistic Regression AUC :", log_acc)
    print("Dummy AUC :", dummy_acc)

    results_table.append({
    "Participant": test_subject,
    "LDA_AUC": lda_acc,
    "SVM_AUC": svm_acc,
    "RandomForest_AUC": rf_acc,
    "XGBoost_AUC": xgb_acc,
    "Logistic_AUC": log_acc,
    "Dummy_AUC": dummy_acc
        })

    #Saving accuracy
    lda_scores.append(lda_acc)
    svm_scores.append(svm_acc)
    rf_scores.append(rf_acc)
    xgb_scores.append(xgb_acc)
    log_scores.append(log_acc)
    dummy_scores.append(dummy_acc)

#Final Results
print("\n Final Results")
print("\nAverage LDA AUC :", np.mean(lda_scores))
print("\nAverage SVM AUC :", np.mean(svm_scores))
print("\nAverage Random Forest AUC :", np.mean(rf_scores))
print("\nAverage XGBoost AUC :", np.mean(xgb_scores))
print("\nAverage Logistic AUC :", np.mean(log_scores))
print("\nAverage Dummy AUC :", np.mean(dummy_scores))

#Saving results in a CSV Table
results_df = pd.DataFrame(results_table)
average_row = {
    "Participant": "Average",
    "LDA_AUC": np.mean(lda_scores),
    "SVM_AUC": np.mean(svm_scores),
    "RandomForest_AUC": np.mean(rf_scores),
    "XGBoost_AUC": np.mean(xgb_scores),
    "Logistic_AUC": np.mean(log_scores),
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
    "Logistic Regression": log_probs,
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
plt.title("Leave-One-Subject-Out ROC Curve")
plt.legend()
plt.grid(True)

#Save the results
plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()
print("ROC curve saved")

#time to run program being printed -----------------------------------------
end_time = time.time()
runtime = end_time - start_time
print(f"\nTotal Runtime: {runtime:.2f} seconds")
print(f"Total Runtime: {runtime/60:.2f} minutes")