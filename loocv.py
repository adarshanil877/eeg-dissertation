#All 12 Participants
import mne
mne.set_log_level("ERROR")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #Importing Scaling Methods
from sklearn.decomposition import PCA #Importing PCA
import matplotlib.pyplot as plt
import numpy as np

#Importing Models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

#Importing Evaluation Metrics
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay

#For file handling
import glob


files = sorted(glob.glob("data/*.set"))

def get_features_labels(file_path):
    
    #Reading the data from EEG data using MNE
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    
    #Filtering between 1-40 hz to remove Low frequencies and Muscle Noises
    raw = raw.copy().filter(1, 40)

    #Creating events based on conditions present on data
    events, event_id = mne.events_from_annotations(raw)

    #Creating epochs - 1 second of data from start of condition
    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin=0,
        tmax=1,
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

all_X = []
all_y = []

print("\nLoading datasets...\n")

for f in files:
    X, y = get_features_labels(f)
    all_X.append(X)
    all_y.append(y)


lda_scores = []
svm_scores = []

print("\nStarting LOOCV...\n")

for i in range(len(all_X)):

    print("Testing Subject", i + 1, "of", len(all_X))

    X_test = all_X[i]
    y_test = all_y[i]

    X_train = np.concatenate([all_X[j] for j in range(len(all_X)) if j != i])
    y_train = np.concatenate([all_y[j] for j in range(len(all_y)) if j != i])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    lda_pred = lda.predict(X_test)
    lda_acc = accuracy_score(y_test, lda_pred)

    svm = SVC(kernel="linear", random_state=42)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_pred)

    print("LDA:", lda_acc, "| SVM:", svm_acc)

    lda_scores.append(lda_acc)
    svm_scores.append(svm_acc)


print("\nLOOCV Results -----------")
print("LDA Mean Accuracy:", np.mean(lda_scores))
print("SVM Mean Accuracy:", np.mean(svm_scores))
print("LDA Std Dev:", np.std(lda_scores))
print("SVM Std Dev:", np.std(svm_scores))