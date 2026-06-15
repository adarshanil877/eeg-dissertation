import mne

file_path = "data/c01_cleaned.set"  

raw = mne.io.read_raw_eeglab(file_path, preload=True)

print(raw)

raw.plot(block=True)