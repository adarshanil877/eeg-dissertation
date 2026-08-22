#ALL PARTICIPANTS INDIVIDUAL TESTING 
import mne
import time

from sklearn.cluster import KMeans
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
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, auc, roc_curve, silhouette_score

#IMPORTS FOR FILE HANDLING
import glob
import os

mne.set_log_level("ERROR")
files = sorted(glob.glob("data/*.set"))

# FOLDER TO SAVE CLUSTERING RESULTS
clustering_folder = "clustering_results"
os.makedirs(clustering_folder, exist_ok=True)

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

def find_best_k(X_scaled):

    #TESTING DIFFERENT CLUSTERS
    silhouette_scores = {}

    print("\nTesting different numbers of clusters")

    for k in range(2, 6):

        #KMEANS MODEL IS CREATED
        model = KMeans(n_clusters=k, random_state=42, n_init=10)

        #MODEL TRAINED TO GET CLUSTERS
        cluster_labels = model.fit_predict(X_scaled)

        #SILHOUTE SCORE CALCULATION
        score = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores[k] = score

        print("K =", k, "Silhouette Score =", round(score, 4))

    #K SELECTED BASED ON BEST SILHOUETTE SCORE
    best_k = max(silhouette_scores, key=silhouette_scores.get)

    print("BEST K :", best_k)
    print("BEST SILHOUETTE SCORE :", round(silhouette_scores[best_k], 4))

    return best_k, silhouette_scores

def run_clustering(X, y, participant):

    print("\nCLUSTERING PARTICIPANT : ", participant)

    #PARTICIPANT FOLDER IS CREATED
    participant_folder = os.path.join(clustering_folder, participant)
    os.makedirs(participant_folder, exist_ok=True)

    #SCALING THE FEATURES
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #FINDING BEST NUMBER OF CLUSTERS
    best_k, silhouette_scores = find_best_k(X_scaled)

    #KMEANS FINAL MODEL
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)

    #CLUSTER ASSIGNMENTS
    cluster_labels = kmeans.fit_predict(X_scaled)

    #CALCULATING FINAL SILHOUETTE SCORE
    final_silhouette = silhouette_score(X_scaled, cluster_labels)

    print("FINAL SILHOUETTE SCORE :", round(final_silhouette, 4))

    print("\nCLUSTER SIZES")
    for cluster in range(best_k):
        cluster_count = np.sum(cluster_labels == cluster)
        print("Cluster", cluster, ":", cluster_count, "epochs")

    #PCA ONLY FOR VISUALISATION
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    #PERCENTAGE OF VARIANCE EXPLAINED
    explained_variance = pca.explained_variance_ratio_ * 100
    print("\nPCA VARIANCE EXPLAINED :", round(explained_variance[0], 2), "%" , "+" , round(explained_variance[1], 2), "%")

    #CLUSTER GROUP CREATED
    plt.figure(figsize=(9, 7))

    for cluster in range(best_k):

        cluster_points = cluster_labels == cluster

        plt.scatter(
            X_pca[cluster_points, 0],
            X_pca[cluster_points, 1],
            label="Cluster " + str(cluster),
            alpha=0.7
        )

    #CLUSTER CENTERS PLOTTED
    cluster_centres_pca = pca.transform(kmeans.cluster_centers_)

    plt.scatter(
        cluster_centres_pca[:, 0],
        cluster_centres_pca[:, 1],
        marker="X",
        s=200,
        label="Cluster Centre"
    )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(participant + " EEG Feature Clustering")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    #GRAPH SAVED
    graph_path = os.path.join(
        participant_folder,
        participant + "_clustering.png"
    )

    plt.savefig(graph_path, dpi=300)
    plt.close()

    print("Clustering graph saved:", graph_path)

    #CSV CREATED
    cluster_data = pd.DataFrame()

    cluster_data["Epoch"] = np.arange(len(y))
    cluster_data["True_Label"] = y
    cluster_data["Cluster"] = cluster_labels
    cluster_data["PCA_1"] = X_pca[:, 0]
    cluster_data["PCA_2"] = X_pca[:, 1]

    #CSV IS SAVED
    csv_path = os.path.join(
        participant_folder,
        participant + "_clusters.csv"
    )

    cluster_data.to_csv(csv_path, index=False)

    print("Cluster data saved:", csv_path)

    #COMPARING CLUSTERS WITH TRUE LABLES
    print("\nCLUSTER VS TRUE LABEL")
    cluster_table = pd.crosstab(cluster_labels, y)
    print(cluster_table)

    return best_k, final_silhouette, cluster_labels, silhouette_scores

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

print("\nRUNNING CLUSTERING FOR ALL PARTICIPANTS")

clustering_results = []

for file in files:

    participant = os.path.basename(file).replace("_cleaned.set", "")

    print("\nPARTICIPANT : ", participant)

    #FEATURES EXTRACTED
    X, y = get_features_labels(file)

    #RUNNIGN THE CLUSTERING
    best_k, silhouette, cluster_labels, silhouette_scores = run_clustering( X, y, participant)

    #SAVING SUMMARY RESULTS
    clustering_results.append({ "Participant": participant, "Best_K": best_k, "Silhouette_Score": silhouette})

#FINAL CLUSTERING RESULTS
summary = pd.DataFrame(clustering_results)
print("\nFINAL CLUSTERING RESULTS")
print(summary)

#SAVING SUMMARY CSV
summary_path = os.path.join(clustering_folder, "clustering_summary.csv")

summary.to_csv(summary_path, index=False)

print("\nSummary saved:", summary_path)

#TIME TO RUN PROGRAM
end_time = time.time()
runtime = end_time - start_time
print(f"\nTOTAL RUNTIME (SECS): {runtime:.2f} seconds")
print(f"TOTAL RUNTIME (MINS): {runtime/60:.2f} minutes")