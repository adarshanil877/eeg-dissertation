import mne
import glob
import os

mne.set_log_level("ERROR")

files = sorted(glob.glob("data/*.set"))

file = files[0]

print("Using participant:", os.path.basename(file))

raw = mne.io.read_raw_eeglab(
    file,
    preload=False
)

events, event_id = mne.events_from_annotations(raw)

print("\nEVENT ID:")
print(event_id)

print("\nFIRST 100 EVENTS:")
print("--------------------------------")

reverse_event_id = {
    value: key
    for key, value in event_id.items()
}

for i in range(min(100, len(events))):

    sample = events[i][0]
    event_number = events[i][2]

    time = sample / raw.info["sfreq"]

    event_name = reverse_event_id.get(
        event_number,
        "UNKNOWN"
    )

    print(
        i,
        "Time:", round(time, 3),
        "Event:", event_name
    )