#Filtering and ML Classifier
import mne
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Importing Models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

#Importing Evaluation Metrics
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay

#Reading the file and convering using MNE
file_path = "data/c01_cleaned.set"
raw = mne.io.read_raw_eeglab(file_path, preload=True)

#Filtering to remove low frequencecies and Muscle noises
raw_filtered = raw.copy().filter(l_freq=1, h_freq=40)

#print(raw_filtered)
#print(raw_filtered.ch_names)

#Creating events based on Conditions present within the data
events, event_id = mne.events_from_annotations(raw_filtered)

#print(events[:10])
#print(event_id)

#creating epochs - 1 second of data from start of condition
epochs = mne.Epochs(
    raw_filtered,
    events,
    event_id,
    tmin=0,
    tmax=1,
    baseline=None,
    preload=True
)
# Computing PSD (Power Spectral Density) - 5 frequency bands and Power
psds = epochs.compute_psd( method="welch",fmin=1,fmax=40)

#X = epochs.get_data() #This is without adding PSD
X = psds.get_data() # After adding PSD
y = epochs.events[:, -1]

print("PSD shape:", X.shape)
print("Labels shape:", y.shape)
#End of Epoch and preprocessing data


# flatten EEG (ML needs 2D)
X = X.reshape(len(X), -1)
print("Flattened PSD shape:", X.shape)

#Creating training and text spilts (80-train 20-test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Normalising and Feature Scaling using meand and sd
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#LDA Model
#model = LinearDiscriminantAnalysis()

#SVM Model
model = SVC(kernel="linear", random_state=42)

#training the model
model.fit(X_train, y_train)

#predicting the Y labels
y_pred = model.predict(X_test)

#accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

#confusion matrix - displaying
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="plasma")
plt.title("Confusion Matrix")
plt.show(block=True)
