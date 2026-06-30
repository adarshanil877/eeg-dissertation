#All 12 Participants
import mne
mne.set_log_level("ERROR")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #Importing Scaling Methods
from sklearn.decomposition import PCA #Importing PCA
import matplotlib.pyplot as plt
import numpy as np
import random

#Importing Models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

#Importing Evaluation Metrics
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay

#For file handling
import glob
import os


files = sorted(glob.glob("data/*.set"))

def preprocess(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, X_test

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
    tmin=-1,
    tmax=0,
    baseline=None,
    preload=True
    )

    # Computing PSD (Power Spectral Density) - 5 frequency bands and Power
    psds = epochs.compute_psd(method="welch", fmin=1, fmax=40)
    X = psds.get_data()
    y = epochs.events[:, -1]

    #Flattening the EEG data (ML Models need 2D data)
    X = X.reshape(len(X), -1)

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

#Main Code - Body

#Dictionary to store all participants
participants = {}

#Reading every participant
for file in files:

    print("\nProcessing:", file)

    X, y = get_features_labels(file)

    participant_name = os.path.basename(file).replace("_cleaned.set", "")

    participants[participant_name] = {
        "X": X,
        "y": y
    }

    print(
        participant_name,
        " X Shape:", X.shape,
        " Y Shape:", y.shape
    )

#Randomly Splitting Participants into Training and Testing Groups
participant_names = list(participants.keys())
random.seed(42)
random.shuffle(participant_names)

#Spliting 80% of data for data
split = int(len(participant_names) * 0.8)

#80% Participants for Training
train_subjects = participant_names[:split]

#Remaining 20% Participants for Testing
test_subjects = participant_names[split:]

print("\nTraining Participants:", train_subjects)
print("Testing Participants:", test_subjects)

#Combining only the Training Participants
X_train = []
y_train = []

for participant in train_subjects:
    X_train.append(participants[participant]["X"])
    y_train.append(participants[participant]["y"])

X_train = np.vstack(X_train)
y_train = np.concatenate(y_train)

#Combining only the Testing Participants
X_test = []
y_test = []

for participant in test_subjects:
    X_test.append(participants[participant]["X"])
    y_test.append(participants[participant]["y"])

X_test = np.vstack(X_test)
y_test = np.concatenate(y_test)

print("\nTraining Dataset")
print("X Shape:", X_train.shape)
print("Y Shape:", y_train.shape)

print("\nTesting Dataset")
print("X Shape:", X_test.shape)
print("Y Shape:", y_test.shape)

#Running LDA
lda_acc = run_lda(X_train, X_test, y_train, y_test)

#Running SVM
svm_acc = run_svm(X_train, X_test, y_train, y_test)

print("\nFINAL RESULTS")
print("LDA Accuracy :", lda_acc)
print("SVM Accuracy :", svm_acc)