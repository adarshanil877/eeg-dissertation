#Just Preprocessing
import mne

file_path = "data/c01_cleaned.set"

raw = mne.io.read_raw_eeglab(file_path, preload=True)
print(raw.info)
print(raw.ch_names)

raw.plot(duration=10, n_channels=10, block=True)