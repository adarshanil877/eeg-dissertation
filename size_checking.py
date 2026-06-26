import mne
import glob
import numpy as np

files = sorted(glob.glob("data/*.set"))

reference_shape = None

for file in files:
    print("\nProcessing:", file)

    raw = mne.io.read_raw_eeglab(file, preload=True)

    raw = raw.copy().filter(1, 40)

    events, event_id = mne.events_from_annotations(raw)

    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin=0,
        tmax=1,
        baseline=None,
        preload=True
    )

    psds = epochs.compute_psd(method="welch", fmin=1, fmax=40)

    X = psds.get_data()
    y = epochs.events[:, -1]

    X = X.reshape(len(X), -1)

    print("X shape:", X.shape)
    print("y shape:", y.shape)