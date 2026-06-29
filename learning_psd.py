import mne
import matplotlib.pyplot as plt
mne.set_log_level("ERROR")

file = "data/c04_cleaned.set"

raw = mne.io.read_raw_eeglab(file, preload=True)

raw = raw.copy().filter(1,40)

raw.pick(["Fz"])

events, event_id = mne.events_from_annotations(raw)

epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id["condition 1"],
    tmin=-1,
    tmax=0,
    baseline=None,
    preload=True
)

epoch = epochs.get_data()[0]

plt.figure(figsize=(12,4))
plt.plot(epoch[0])
plt.title("Raw EEG (Time Domain) - Fz")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()

psds = epochs.compute_psd(
    method="welch",
    fmin=1,
    fmax=40
)

fig = psds.plot()
plt.show()

print(psds.get_data().shape)

print(psds.freqs)

print(psds.get_data()[0, 0])