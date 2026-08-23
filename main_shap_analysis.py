#ALL PARTICIPANTS SHAP ANALYSIS

import mne
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import glob
import os

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from mne_features.feature_extraction import extract_features
from xgboost import XGBClassifier

mne.set_log_level("ERROR")

files = sorted(glob.glob("data/*.set"))

#CREATING SHAP OUTPUT FOLDER
shap_folder = "shap_analysis"
os.makedirs(shap_folder, exist_ok=True)

def get_psd_features(epochs):

    #COMPUTING PSD FOR ALL FREQUENCIES
    psds = epochs.compute_psd(method="welch", fmin=1, fmax=40)

    psd_features = psds.get_data()

    #FLATTENING EEG DATA
    psd_features = psd_features.reshape(len(psd_features), -1)

    return psds, psd_features

def get_mne_features(epochs):

    #GETTING EEG SIGNAL DATA
    data = epochs.get_data()

    #EXTRACTING MNE FEATURES
    features = extract_features(data, sfreq=epochs.info["sfreq"], selected_funcs=["line_length","kurtosis","skewness","hjorth_mobility","hjorth_complexity","zero_crossings","spect_entropy","svd_entropy","app_entropy","samp_entropy"])

    return features

def get_features_labels(file_path):

    #READING EEG DATA
    raw = mne.io.read_raw_eeglab(file_path, preload=True)

    #FILTERING BETWEEN 1-40 HZ
    raw = raw.copy().filter(1, 40)

    #SELECTING FRONTAL EEG CHANNELS
    channels = ["Fz","FCz","F2","F3","F4","F6"]
    raw = raw.pick(channels)

    #CREATING EVENTS
    events, event_id = mne.events_from_annotations(raw)

    #CREATING EPOCHS
    epochs = mne.Epochs(raw, events, event_id={"condition 1":event_id["condition 1"],"condition 2":event_id["condition 2"]}, tmin=-0.9, tmax=0, baseline=None, preload=True)

    #GETTING PSD FEATURES
    psds, psd_features = get_psd_features(epochs)

    #GETTING MNE FEATURES
    mne_features = get_mne_features(epochs)

    #COMBINING PSD AND MNE FEATURES
    X = np.concatenate((psd_features,mne_features),axis=1)

    #CREATING LABELS
    encoder = LabelEncoder()
    y = encoder.fit_transform(epochs.events[:,-1])

    #CREATING FEATURE NAMES
    feature_names = []

    #CREATING PSD FEATURE NAMES
    freqs = psds.freqs

    for channel in channels:
        for frequency in freqs:
            feature_names.append("PSD_"+channel+"_"+str(round(frequency,1))+"Hz")

    #CREATING MNE FEATURE NAMES
    mne_feature_names = ["line_length","kurtosis","skewness","hjorth_mobility","hjorth_complexity","zero_crossings","spect_entropy","svd_entropy","app_entropy","samp_entropy"]

    for feature in mne_feature_names:
        for channel in channels:
            feature_names.append(channel+"_"+feature)

    print("FEATURE SHAPE :",X.shape)

    return X,y,feature_names

def run_shap_analysis(X,y,feature_names,participant):

    #SCALING ALL PARTICIPANT DATA FOR SHAP ANALYSIS
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #TRAINING XGBOOST MODEL
    model = XGBClassifier(n_estimators=100,max_depth=6,learning_rate=0.1,random_state=42,eval_metric="logloss")
    model.fit(X_scaled,y)

    #CREATING SHAP EXPLAINER
    explainer = shap.TreeExplainer(model)

    #CALCULATING SHAP VALUES
    shap_values = explainer.shap_values(X_scaled)

    #GETTING BASE VALUE FOR WATERFALL PLOT
    base_value = explainer.expected_value

    if isinstance(base_value,np.ndarray):
        base_value = base_value[0]

    #CALCULATING MEAN ABSOLUTE SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    #CALCULATING MEAN SHAP
    mean_shap = shap_values.mean(axis=0)

    #CREATING SHAP RESULTS DATAFRAME
    shap_results = pd.DataFrame({"PARTICIPANT":participant,"FEATURE":feature_names,"MEAN_ABSOLUTE_SHAP":mean_abs_shap,"MEAN_SHAP":mean_shap})

    #SORTING FEATURES FROM MOST IMPORTANT TO LEAST IMPORTANT
    shap_results = shap_results.sort_values("MEAN_ABSOLUTE_SHAP",ascending=False)

    #SAVING ALL FEATURE SHAP VALUES
    shap_results.to_csv(os.path.join(shap_folder,participant+"_SHAP_FEATURES.csv"),index=False)

    #GETTING TOP 20 FEATURES
    top_20 = shap_results.head(20).copy()

    #GETTING WORST 20 FEATURES
    worst_20 = shap_results.tail(20).copy()

    #SAVING TOP 20 FEATURES
    top_20.to_csv(os.path.join(shap_folder,participant+"_TOP_20_FEATURES.csv"),index=False)

    #SAVING WORST 20 FEATURES
    worst_20.to_csv(os.path.join(shap_folder,participant+"_WORST_20_FEATURES.csv"),index=False)

    #CREATING SHAP EXPLANATION FOR WATERFALL
    sample_index = 0

    shap_explanation = shap.Explanation(values=shap_values[sample_index],base_values=base_value,data=X_scaled[sample_index],feature_names=feature_names)

    #CREATING WATERFALL PLOT
    plt.figure()
    shap.plots.waterfall(shap_explanation,max_display=20,show=False)
    plt.title(participant+" SHAP Waterfall")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_folder,participant+"_WATERFALL.png"),dpi=300,bbox_inches="tight")
    plt.close()

    #CREATING BEESWARM PLOT
    plt.figure()
    shap.summary_plot(shap_values,X_scaled,feature_names=feature_names,max_display=20,show=False)
    plt.title(participant+" SHAP Beeswarm")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_folder,participant+"_BEESWARM.png"),dpi=300,bbox_inches="tight")
    plt.close()

    #CREATING BAR PLOT OF TOP FEATURES
    plt.figure()
    shap.summary_plot(shap_values,X_scaled,feature_names=feature_names,plot_type="bar",max_display=20,show=False)
    plt.title(participant+" SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_folder,participant+"_FEATURE_IMPORTANCE.png"),dpi=300,bbox_inches="tight")
    plt.close()

    #PRINTING TOP FEATURES
    print("\nTOP 20 FEATURES FOR",participant)

    for _,row in top_20.iterrows():
        print(row["FEATURE"],":",round(row["MEAN_ABSOLUTE_SHAP"],4))

    #PRINTING WORST FEATURES
    print("\nWORST 20 FEATURES FOR",participant)

    for _,row in worst_20.iterrows():
        print(row["FEATURE"],":",round(row["MEAN_ABSOLUTE_SHAP"],4))

    return shap_results

#MAIN BODY

start_time = time.time()

print("\nRUNNING SHAP ANALYSIS FOR ALL PARTICIPANTS")

all_shap_results = []

for file in files:

    #GETTING PARTICIPANT NAME
    participant = os.path.basename(file).replace("_cleaned.set","")

    print("\nPARTICIPANT :",participant)

    #EXTRACTING FEATURES
    X,y,feature_names = get_features_labels(file)

    #RUNNING SHAP ONCE FOR THIS PARTICIPANT
    participant_results = run_shap_analysis(X,y,feature_names,participant)

    #STORING RESULTS
    all_shap_results.append(participant_results)

#COMBINING ALL PARTICIPANT RESULTS
all_shap_results = pd.concat(all_shap_results,ignore_index=True)

#SAVING ALL PARTICIPANT SHAP RESULTS
all_shap_results.to_csv(os.path.join(shap_folder,"ALL_PARTICIPANTS_SHAP_RESULTS.csv"),index=False)

#GETTING TOP 20 FEATURES FOR EACH PARTICIPANT
top_features = all_shap_results.groupby("PARTICIPANT").head(20)

#COUNTING HOW OFTEN EACH FEATURE APPEARS IN TOP 20
feature_frequency = top_features.groupby("FEATURE").size().reset_index(name="TOP_20_COUNT")

#SORTING BY FREQUENCY
feature_frequency = feature_frequency.sort_values("TOP_20_COUNT",ascending=False)

#SAVING FEATURE FREQUENCY
feature_frequency.to_csv(os.path.join(shap_folder,"FEATURE_FREQUENCY_TOP_20.csv"),index=False)

#GETTING WORST 20 FEATURES FOR EACH PARTICIPANT
worst_features = all_shap_results.groupby("PARTICIPANT").tail(20)

#COUNTING HOW OFTEN EACH FEATURE APPEARS IN WORST 20
worst_frequency = worst_features.groupby("FEATURE").size().reset_index(name="WORST_20_COUNT")

#SORTING BY FREQUENCY
worst_frequency = worst_frequency.sort_values("WORST_20_COUNT",ascending=False)

#SAVING WORST FEATURE FREQUENCY
worst_frequency.to_csv(os.path.join(shap_folder,"FEATURE_FREQUENCY_WORST_20.csv"),index=False)

#CREATING OVERALL FEATURE IMPORTANCE
overall_feature_importance = all_shap_results.groupby("FEATURE")["MEAN_ABSOLUTE_SHAP"].mean().reset_index()

#SORTING OVERALL FEATURE IMPORTANCE
overall_feature_importance = overall_feature_importance.sort_values("MEAN_ABSOLUTE_SHAP",ascending=False)

#SAVING OVERALL FEATURE IMPORTANCE
overall_feature_importance.to_csv(os.path.join(shap_folder,"OVERALL_FEATURE_IMPORTANCE.csv"),index=False)

#PRINTING MOST COMMON FEATURES
print("\nMOST COMMON TOP FEATURES")

print(feature_frequency.head(20).to_string(index=False))

#PRINTING MOST COMMON WORST FEATURES
print("\nMOST COMMON WORST FEATURES")

print(worst_frequency.head(20).to_string(index=False))

#PRINTING OVERALL FEATURE IMPORTANCE
print("\nOVERALL FEATURE IMPORTANCE")

print(overall_feature_importance.head(20).to_string(index=False))

#TIME TO RUN PROGRAM
end_time = time.time()

runtime = end_time-start_time

print(f"\nTOTAL RUNTIME (SECS): {runtime:.2f} seconds")
print(f"TOTAL RUNTIME (MINS): {runtime/60:.2f} minutes")