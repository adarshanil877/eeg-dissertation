import mne
mne.set_log_level("ERROR")

import glob

files = sorted(glob.glob("data/*.set"))

for file in files:

    print("\n====================================")
    print("FILE:", file)
    print("====================================")

    raw = mne.io.read_raw_eeglab(file, preload=False)

    events, event_id = mne.events_from_annotations(raw)

    print("\nEvent IDs:")
    print(event_id)

    print("\nNumber of each event:")

    for name, code in event_id.items():

        count = (events[:, 2] == code).sum()

        print(name, "-> Code:", code, " Count:", count)