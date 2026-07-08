#All 12 Participants
import mne
mne.set_log_level("ERROR")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #Importing Scaling Methods
from sklearn.decomposition import PCA #Importing PCA
import matplotlib.pyplot as plt
import numpy as np
import random 
from sklearn.preprocessing import LabelEncoder

#Importing Models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#Importing Evaluation Metrics
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay

#For file handling
import glob
import os


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

def get_hjorth_features(epochs):

    #Getting EEG signal data
    data = epochs.get_data()

    hjorth_features = []

    #Looping through every epoch
    for epoch in data:

        epoch_features = []

        #Looping through every EEG channel
        for channel in epoch:

            #First derivative - change in signal
            first_derivative = np.diff(channel)

            #Second derivative - change in the first derivative
            second_derivative = np.diff(first_derivative)

            #Computing Hjorth Activity - The fluctuation of signal
            activity = np.var(channel)

            #Hjorth Mobility - How slow or fast the waves are
            mobility = np.sqrt(
                np.var(first_derivative) / activity
            )

            #Hjorth Complexity - how smooth or irregular the waves are
            complexity = (
                np.sqrt(
                    np.var(second_derivative) /
                    np.var(first_derivative)
                )
                / mobility
            )

            #Adding 3 Hjorth values for each channel
            epoch_features.extend([
                activity,
                mobility,
                complexity
            ])

        hjorth_features.append(epoch_features)

    return np.array(hjorth_features)

def get_bandpower_features(psds):

    #Get individual frequencies in bandpower
    freqs = psds.freqs

    #Getting all the PSD values
    psd_data = psds.get_data()

    #Defining the Bands to be used in Bandpower features - Standard EEG bands
    bands = {
        "delta": (1,4),
        "theta": (4,8),
        "alpha": (8,13),
        "beta": (13,30),
        "gamma": (30,40)
    }

    bandpower = []

    #Calculating the average value for bandpower - Looping through every band's high and low values
    for low, high in bands.values():
        
        idx = (freqs >= low) & (freqs < high)
        
        #Mean power inside each frequency band for which the condition of IDX is true
        band = psd_data[:, :, idx].mean(axis=2)
        
        #Appending the power as per band into pandpower
        bandpower.append(band)

    bandpower = np.concatenate(bandpower, axis=1)
    return bandpower

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
    y = epochs.events[:, -1]
    
    # Using only PSD (Experiment 1)
    X = psd_features

    #Getting Bandpower Features (Experiment 2)
    #bandpower_features = get_bandpower_features(psds)
    #Combining PSD + Bandpower
    #X = np.concatenate((psd_features, bandpower_features), axis=1)
    
    #Getting Hjorth Features (Experiment 3)
    #hjorth_features = get_hjorth_features(epochs)
    #Combining PSD + Hjorth Features
    #X = np.concatenate((psd_features, hjorth_features), axis=1)
    
    #Getting CSP Features (Experiment 4 - Experimental and wrong)
    #csp_features = get_csp_features(epochs, y)
    #Combining PSD + CSP
    #X = np.concatenate((psd_features,csp_features), axis=1)
    
    #print("Feature Shape:", X.shape)

    return X, y

#FUNCTION TO RUN THE LDA MODEL----------------------
def run_lda(X_train, X_test, y_train, y_test):

    #Calling  Preprocessing Function
    X_train, X_test = preprocess(X_train, X_test)

    #MODEL 1: LINEAR DISCRIMINANT ANALYSIS (LDA)
    model = LinearDiscriminantAnalysis()
    
    #Training the Model
    model.fit(X_train, y_train)

    #Predicting the Y Labels on the Test Dataset
    y_pred = model.predict(X_test)

    #Accuracy
    return accuracy_score(y_test, y_pred)

#FUNCTION TO RUN THE SVM MODEL----------------------
def run_svm(X_train, X_test, y_train, y_test):

    #Calling Preprocessing function
    X_train, X_test = preprocess(X_train, X_test)

    #MODEL 2: SUPPORT VECTOR MACHINE (SVM)
    model = SVC(kernel="linear", random_state=42)
    
    #Training the Model
    model.fit(X_train, y_train)

    #Predicting the Y Labels on Test Dataset
    y_pred = model.predict(X_test)

    #Accuracy
    return accuracy_score(y_test, y_pred)

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
    y_pred = model.predict(X_test)

    #Accuracy
    return accuracy_score(y_test, y_pred)

#FUNCTION TO RUN THE XGBOOST MODEL----------------------
def run_xgboost(X_train, X_test, y_train, y_test):

    #Calling Preprocessing Function
    X_train, X_test = preprocess(X_train, X_test)

    #Encode labels as 0 and 1
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

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
    y_pred = model.predict(X_test)

    #Accuracy
    return accuracy_score(y_test, y_pred)


#Main Code - Body

#Dictionary to store all participants
participants = {}

#Reading every participant
for file in files:

    X, y = get_features_labels(file)

    #Using participant names from filenames
    participant_name = os.path.basename(file).replace("_cleaned.set", "")

    #Adding the features to dictionary
    participants[participant_name] = {
        "X": X,
        "y": y
    }

lda_scores = []
svm_scores = []
rf_scores =[]
xgb_scores = []

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
    X_train = []
    y_train = []

    for participant in train_subjects:
        X_train.append(participants[participant]["X"])
        y_train.append(participants[participant]["y"])

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)

    #Testing Participant
    X_test = participants[test_subject]["X"]
    y_test = participants[test_subject]["y"]

    #Running LDA
    lda_acc = run_lda(X_train, X_test, y_train, y_test)

    #Running SVM
    svm_acc = run_svm(X_train, X_test, y_train, y_test)

    #Running Random Forest
    rf_acc = run_random_forest(X_train, X_test, y_train, y_test)

    #Running XGBoost
    xgb_acc = run_xgboost(X_train, X_test, y_train, y_test)

    print("\nLDA Accuracy :", lda_acc)
    print("SVM Accuracy :", svm_acc)
    print("Random Forest Accuracy :", rf_acc)
    print("XGBoost Accuracy :", xgb_acc)

    #Saving accuracy
    lda_scores.append(lda_acc)
    svm_scores.append(svm_acc)
    rf_scores.append(rf_acc)
    xgb_scores.append(xgb_acc)

#Final Results
print("\n Final Results")
print("\nAverage LDA Accuracy :", np.mean(lda_scores))
print("\nAverage SVM Accuracy :", np.mean(svm_scores))
print("\nAverage Random Forest Accuracy :", np.mean(rf_scores))
print("\nAverage XGBoost Accuracy :", np.mean(xgb_scores))