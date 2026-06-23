#All 12 Participants
import mne
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #Importing Scaling Methods
from sklearn.decomposition import PCA #Importing PCA
import matplotlib.pyplot as plt

#Importing Models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

#Importing Evaluation Metrics
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay

#For file handling
import glob


files = sorted(glob.glob("data/*.set"))
print(files)

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

#FUNCTION TO RUN THE LDA MODEL----------------------
def run_lda(X, y):

    #Train and Test splits are done with 80-Train and 20-Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    #Feature Scaling using Mean and Standard Deviation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Doing Principal Component Analysis to reduce dimensionality
    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    #MODEL 1: LINEAR DISCRIMINANT ANALYSIS (LDA)
    model = LinearDiscriminantAnalysis()
    
    #Training the Model
    model.fit(X_train, y_train)

    #Predicting the Y Labels on the Test Dataset
    y_pred = model.predict(X_test)

    #Accuracy
    return accuracy_score(y_test, y_pred)

#FUNCTION TO RUN THE SVM MODEL----------------------
def run_svm(X, y):
    
    #Train and Test splits are done with 80-Train and 20-Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    #Feature Scaling using Mean and Standard Deviation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Doing Principal Component Analysis to reduce dimensionality
    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    #MODEL 2: SUPPORT VECTOR MACHINE (SVM)
    model = SVC(kernel="linear", random_state=42)
    
    #Training the Model
    model.fit(X_train, y_train)

    #Predicting the Y Labels on Test Dataset
    y_pred = model.predict(X_test)

    #Accuracy
    return accuracy_score(y_test, y_pred)

#MAIN CODE ------------------------------------

#Arrays to store the Accuracy scores from both models for each Participant
lda_scores = []
svm_scores = []

#Looping through each participant to find Accuracy score
for file in files:
    
    #Printing the file being processes
    print("Processing:", file)

    #Calling function to Preprocess data and returning the preprocessed data
    X, y = get_features_labels(file)

    #Running the LDA MODEL
    lda_acc = run_lda(X, y)

    #Running the SVM MODEL
    svm_acc = run_svm(X, y)

    #Printing the accuracy for both models for each person at end of 1 iteration
    print("LDA:", lda_acc, "SVM:", svm_acc)

    #Adding score to the Array so that we can calculate average accuracy
    lda_scores.append(lda_acc)
    svm_scores.append(svm_acc)

#Final results of both the Models 
print("\nFinal Results")
print("LDA Mean:", sum(lda_scores)/len(lda_scores))
print("SVM Mean:", sum(svm_scores)/len(svm_scores))