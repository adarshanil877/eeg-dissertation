import mne
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

file_path = "data/c01_cleaned.set"

raw = mne.io.read_raw_eeglab(file_path, preload=True)

# Step 1: basic filtering
raw_filtered = raw.copy().filter(l_freq=1, h_freq=40)

print(raw_filtered)
print(raw_filtered.ch_names)

events, event_id = mne.events_from_annotations(raw_filtered)

print(events[:10])
print(event_id)

epochs = mne.Epochs(
    raw_filtered,
    events,
    event_id,
    tmin=0,
    tmax=1,
    baseline=None,
    preload=True
)
X = epochs.get_data()
y = epochs.events[:, -1]

print("X shape:", X.shape)
print("y shape:", y.shape)
#End of Epoch and preprocessing data


# flatten EEG (ML needs 2D)
X = X.reshape(X.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))