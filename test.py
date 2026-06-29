import mne
import glob
import numpy as np

mne.set_log_level("ERROR")

files = sorted(glob.glob("data/*.set"))

def inspect_file(file_path):

    print("\n" + "="*50)
    print("FILE:", file_path)
    print("="*50)

    # Load raw EEG
    raw = mne.io.read_raw_eeglab(file_path, preload=True)

    # 1. Channels
    print("\nCHANNEL INFO")
    print("Number of channels:", len(raw.ch_names))
    print("Channel names:", raw.ch_names)

    # 2. Events
    events, event_id = mne.events_from_annotations(raw)

    print("\nEVENT INFO")
    print("Event mapping:", event_id)
    print("Total events:", len(events))

    # 3. Epochs
    try:
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=-1,
            tmax=0,
            baseline=None,
            preload=True,
            verbose=False
        )

        print("\nEPOCH INFO")
        print("Epochs created:", len(epochs))

    except Exception as e:
        print("\nEPOCH ERROR:", e)


# Run for all participants
for f in files:
    inspect_file(f)