#SHAP ANALYSIS FOR ALL PARTICIPANTS

import mne
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mne_features.feature_extraction import extract_features
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import shap
import pandas as pd
import glob
import os

mne.set_log_level("ERROR")
files = sorted(glob.glob("data/*.set"))

#CREATING SHAP RESULTS FOLDER
shap_folder = "SHAP_RESULTS"
os.makedirs(shap_folder, exist_ok=True)

def get_psd_features(epochs):
    #COMPUTING PSD
    psds = epochs.compute_psd(method="welch", fmin=1, fmax=40)
    psd_features = psds.get_data()
    psd_features = psd_features.reshape(len(psd_features), -1)
    return psds, psd_features

def get_mne_features(epochs):
    #GETTING EEG SIGNAL DATA
    data = epochs.get_data()
    #EXTRACTING MNE FEATURES
    features = extract_features(data, sfreq=epochs.info["sfreq"], selected_funcs=["line_length", "kurtosis", "skewness", "hjorth_mobility", "hjorth_complexity", "zero_crossings", "spect_entropy", "svd_entropy", "app_entropy", "samp_entropy"])
    return features

def get_features_labels(file_path):
    #READING EEG DATA
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    #FILTERING EEG DATA
    raw = raw.copy().filter(1, 40)
    #SELECTING FRONTAL CHANNELS
    channels = ["Fz", "FCz", "F2", "F3", "F4", "F6"]
    raw = raw.pick(channels)
    #CREATING EVENTS
    events, event_id = mne.events_from_annotations(raw)
    #CREATING EPOCHS
    epochs = mne.Epochs(raw, events, event_id={"condition 1": event_id["condition 1"], "condition 2": event_id["condition 2"]}, tmin=-0.9, tmax=0, baseline=None, preload=True)
    #GETTING PSD FEATURES
    psds, psd_features = get_psd_features(epochs)
    #GETTING MNE FEATURES
    mne_features = get_mne_features(epochs)
    #COMBINING FEATURES
    X = np.concatenate((psd_features, mne_features), axis=1)
    #CREATING LABELS
    encoder = LabelEncoder()
    y = encoder.fit_transform(epochs.events[:, -1])
    #CREATING FEATURE NAMES
    feature_names = []
    freqs = psds.freqs
    for channel in channels:
        for frequency in freqs:
            feature_names.append("PSD_" + channel + "_" + str(round(frequency, 1)) + "Hz")
    mne_feature_names = ["line_length", "kurtosis", "skewness", "hjorth_mobility", "hjorth_complexity", "zero_crossings", "spect_entropy", "svd_entropy", "app_entropy", "samp_entropy"]
    for feature in mne_feature_names:
        for channel in channels:
            feature_names.append(channel + "_" + feature)
    print("FEATURE SHAPE :", X.shape)
    return X, y, feature_names

def get_shap_top_features(X_train_scaled, y_train, feature_names, n_features=60):
    #TRAINING XGBOOST FOR FEATURE SELECTION
    model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric="logloss")
    model.fit(X_train_scaled, y_train)
    #CALCULATING SHAP VALUES
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_scaled)
    #CALCULATING MEAN ABSOLUTE SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    #RANKING FEATURES
    ranking = np.argsort(mean_abs_shap)[::-1]
    selected_indices = ranking[:n_features].tolist()
    selected_features = [feature_names[i] for i in selected_indices]
    return selected_indices, selected_features

def run_shap_analysis(X, y, feature_names, participant):
    #CREATING CROSS VALIDATION
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_shap_values = []
    all_test_data = []
    all_feature_names = []
    print("\nSTARTING SHAP ANALYSIS :", participant)

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
        print("FOLD :", fold)

        #SPLITTING DATA
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        #SCALING DATA
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        #SELECTING TOP SHAP FEATURES
        selected_indices, selected_features = get_shap_top_features(X_train_scaled, y_train, feature_names, n_features=60)
        X_train_selected = X_train_scaled[:, selected_indices]
        X_test_selected = X_test_scaled[:, selected_indices]

        #TRAINING XGBOOST
        model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric="logloss")
        model.fit(X_train_selected, y_train)

        #CALCULATING SHAP VALUES ON TEST DATA
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_selected)

        all_shap_values.append(shap_values)
        all_test_data.append(X_test_selected)
        all_feature_names.append(selected_features)

    #COMBINING SHAP VALUES
    combined_shap = np.concatenate(all_shap_values, axis=0)
    combined_data = np.concatenate(all_test_data, axis=0)
    final_feature_names = all_feature_names[0]

    #CREATING SHAP RESULTS
    mean_abs_shap = np.abs(combined_shap).mean(axis=0)
    mean_shap = combined_shap.mean(axis=0)

    shap_results = pd.DataFrame({"FEATURE": final_feature_names, "MEAN_ABSOLUTE_SHAP": mean_abs_shap, "MEAN_SHAP": mean_shap})
    shap_results = shap_results.sort_values("MEAN_ABSOLUTE_SHAP", ascending=False).reset_index(drop=True)
    shap_results.insert(0, "RANK", range(1, len(shap_results) + 1))

    #CREATING PARTICIPANT FOLDER
    participant_folder = os.path.join(shap_folder, participant)
    os.makedirs(participant_folder, exist_ok=True)

    #SAVING SHAP RESULTS
    shap_results.to_csv(os.path.join(participant_folder, participant + "_SHAP_FEATURE_IMPORTANCE.csv"), index=False)

    #SAVING TOP 10 FEATURES
    shap_results.head(10).to_csv(os.path.join(participant_folder, participant + "_TOP_10_FEATURES.csv"), index=False)

    #SAVING BOTTOM 10 FEATURES
    shap_results.tail(10).to_csv(os.path.join(participant_folder, participant + "_BOTTOM_10_FEATURES.csv"), index=False)

    #CREATING SHAP EXPLANATION
    shap_explanation = shap.Explanation(values=combined_shap, data=combined_data, feature_names=final_feature_names)

    #CREATING BEESWARM PLOT
    shap.summary_plot(combined_shap, combined_data, feature_names=final_feature_names, max_display=20, show=False)
    plt.title(participant + " SHAP BEESWARM")
    plt.tight_layout()
    plt.savefig(os.path.join(participant_folder, participant + "_SHAP_BEESWARM.png"), dpi=300, bbox_inches="tight")
    plt.close()

    #SELECTING REPRESENTATIVE SAMPLE FOR WATERFALL
    sample_index = np.argmax(np.abs(combined_shap).sum(axis=1))

    #CREATING WATERFALL PLOT
    shap.plots.waterfall(shap_explanation[sample_index], max_display=20, show=False)
    plt.title(participant + " SHAP WATERFALL")
    plt.tight_layout()
    plt.savefig(os.path.join(participant_folder, participant + "_SHAP_WATERFALL.png"), dpi=300, bbox_inches="tight")
    plt.close()

    #PRINTING TOP FEATURES
    print("\nTOP FEATURES :", participant)
    for i in range(min(10, len(shap_results))):
        print(shap_results.iloc[i]["FEATURE"], ":", round(shap_results.iloc[i]["MEAN_ABSOLUTE_SHAP"], 6))

    return shap_results

#MAIN BODY
start_time = time.time()
print("\nRUNNING SHAP ANALYSIS FOR ALL PARTICIPANTS")

all_participant_results = []

for file in files:
    participant = os.path.basename(file).replace("_cleaned.set", "")
    print("\nPARTICIPANT :", participant)

    #EXTRACTING FEATURES
    X, y, feature_names = get_features_labels(file)

    #RUNNING SHAP ANALYSIS
    participant_results = run_shap_analysis(X, y, feature_names, participant)
    participant_results["PARTICIPANT"] = participant
    all_participant_results.append(participant_results)

#COMBINING ALL PARTICIPANTS
combined_results = pd.concat(all_participant_results, ignore_index=True)

#SAVING ALL PARTICIPANT RESULTS
combined_results.to_csv(os.path.join(shap_folder, "ALL_PARTICIPANTS_SHAP_RESULTS.csv"), index=False)

#GETTING TOP 10 FEATURES FROM ALL PARTICIPANTS
top_10_all = combined_results[combined_results["RANK"] <= 10]

feature_frequency = top_10_all.groupby("FEATURE").agg(PARTICIPANTS_IN_TOP_10=("PARTICIPANT", "nunique"), AVERAGE_SHAP=("MEAN_ABSOLUTE_SHAP", "mean"), AVERAGE_RANK=("RANK", "mean")).reset_index()

feature_frequency = feature_frequency.sort_values(["PARTICIPANTS_IN_TOP_10", "AVERAGE_SHAP"], ascending=[False, False])

feature_frequency.to_csv(os.path.join(shap_folder, "FEATURE_FREQUENCY_TOP_10.csv"), index=False)

#GETTING TOP 20 FEATURES FROM ALL PARTICIPANTS
top_20_all = combined_results[combined_results["RANK"] <= 20]

feature_frequency_20 = top_20_all.groupby("FEATURE").agg(PARTICIPANTS_IN_TOP_20=("PARTICIPANT", "nunique"), AVERAGE_SHAP=("MEAN_ABSOLUTE_SHAP", "mean"), AVERAGE_RANK=("RANK", "mean")).reset_index()
feature_frequency_20 = feature_frequency_20.sort_values(["PARTICIPANTS_IN_TOP_20", "AVERAGE_SHAP"], ascending=[False, False])
feature_frequency_20.to_csv(os.path.join(shap_folder, "FEATURE_FREQUENCY_TOP_20.csv"), index=False)

#AVERAGE FEATURE IMPORTANCE ACROSS PARTICIPANTS
average_feature_importance = combined_results.groupby("FEATURE").agg(AVERAGE_ABSOLUTE_SHAP=("MEAN_ABSOLUTE_SHAP", "mean"), AVERAGE_SHAP=("MEAN_SHAP", "mean"), PARTICIPANTS=("PARTICIPANT", "nunique")).reset_index()
average_feature_importance = average_feature_importance.sort_values("AVERAGE_ABSOLUTE_SHAP", ascending=False)
average_feature_importance.to_csv(os.path.join(shap_folder, "AVERAGE_FEATURE_IMPORTANCE.csv"), index=False)

#PRINTING MOST COMMON FEATURES
print("\nMOST COMMON TOP FEATURES")
print(feature_frequency.head(20).to_string(index=False))

#PRINTING LEAST IMPORTANT FEATURES
print("\nLEAST IMPORTANT FEATURES")
print(average_feature_importance.tail(20).to_string(index=False))

#TIME TO RUN PROGRAM
end_time = time.time()
runtime = end_time - start_time

print(f"\nTOTAL RUNTIME : {runtime:.2f} SECONDS")
print(f"TOTAL RUNTIME : {runtime/60:.2f} MINUTES")