
"""
Cough vs Non-Cough Classification 
"""

import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# =====================
# CONFIG
# =====================
SAMPLE_RATE = 16000
DATASET_ROOT = "."   # current folder (KHacks)

COUGH_FOLDER = "Cough"
NON_COUGH_FOLDER = "Noncough"
MODEL_PATH = "Models/cough_model.joblib"


'''DATASET_ROOT = "Dataset_Cough"   # relative path
MODEL_PATH = "cough_model.joblib"

COUGH_FOLDER = "Cough"
NON_COUGH_FOLDER = "Noncough"
'''

# =====================
# FEATURE EXTRACTION
# =====================
def extract_features(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)

    if len(y) < 1000:  # skip extremely short clips
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    rms = librosa.feature.rms(y=y).mean()

    return np.concatenate([mfcc, [zcr, rms]])

# =====================
# LOAD DATASET
# =====================
X, y = [], []

for label_name, label_value in [(COUGH_FOLDER, 1), (NON_COUGH_FOLDER, 0)]:
    folder_path = os.path.join(DATASET_ROOT, label_name)
    print(f"Scanning: {folder_path}")

    for fname in os.listdir(folder_path):
        if fname.lower().endswith((".wav", ".mp3")):
            fpath = os.path.join(folder_path, fname)
            feats = extract_features(fpath)

            if feats is not None:
                X.append(feats)
                y.append(label_value)

X = np.array(X)
y = np.array(y)

print("\nTotal samples:", len(X))

# =====================
# TRAIN MODEL
# =====================
clf = RandomForestClassifier(
    n_estimators=50,
    random_state=42
)

clf.fit(X, y)

print("\nTraining performance (demo only):")
print(classification_report(
    y,
    clf.predict(X),
    target_names=["Non-Cough", "Cough"]
))

joblib.dump(clf, MODEL_PATH)
print(f"\nModel saved as {MODEL_PATH}")

# =====================
# LOAD MODEL FOR TESTING
# =====================
clf = joblib.load(MODEL_PATH)

def extract_features_simple(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    rms = librosa.feature.rms(y=y).mean()

    return np.concatenate([mfcc, [zcr, rms]])

# =====================
# SINGLE FILE TEST
# =====================
TEST_AUDIO = "test_audio.wav"   # change this if needed

if os.path.exists(TEST_AUDIO):
    feats = extract_features_simple(TEST_AUDIO).reshape(1, -1)
    pred = clf.predict(feats)[0]
    label = "Cough" if pred == 1 else "Non-Cough"
    print(f"\n{TEST_AUDIO} → {label}")
else:
    print("\nNo test audio found. Skipping single-file test.")

# =====================
# COUGH COUNTING
# =====================
def count_coughs_in_audio(path, win_sec=0.5, hop_sec=0.25, prob_thresh=0.7):

    full_feats = extract_features(path).reshape(1, -1)
    probs = clf.predict_proba(full_feats)[0]
    pred = clf.predict(full_feats)[0]

    if pred == 0:
        return "Non-Cough", float(probs[0]), 0

    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    win_len = int(win_sec * sr)
    hop_len = int(hop_sec * sr)

    def extract_segment_features(segment):
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13).mean(axis=1)
        zcr = librosa.feature.zero_crossing_rate(segment).mean()
        rms = librosa.feature.rms(y=segment).mean()
        return np.concatenate([mfcc, [zcr, rms]])

    cough_count = 0

    for start in range(0, len(y) - win_len, hop_len):
        segment = y[start:start + win_len]
        feats = extract_segment_features(segment).reshape(1, -1)

        seg_probs = clf.predict_proba(feats)[0]
        seg_pred = clf.predict(feats)[0]

        if seg_pred == 1 and seg_probs[1] >= prob_thresh:
            cough_count += 1

    return "Cough", float(probs[1]), cough_count

# =====================
# COUNT COUGHS IN FILE
# =====================
if os.path.exists(TEST_AUDIO):
    label, confidence, n_coughs = count_coughs_in_audio(TEST_AUDIO)

    print("\nFinal analysis:")
    print("Label:", label)
    print("Confidence:", confidence)
    print("Cough count inside file:", n_coughs)

def analyze_audio(file_path):
    """
    The function app.py will call.
    It returns the Label, Confidence, and Count.
    """
    if not os.path.exists(MODEL_PATH):
        return "Model Missing", 0, 0
    
    # Use your existing logic here
    label, confidence, n_coughs = count_coughs_in_audio(file_path)
    return label, confidence, n_coughs