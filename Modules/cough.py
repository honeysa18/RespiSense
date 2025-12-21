import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- 1. CONFIGURATION ---
SAMPLE_RATE = 16000
DATASET_ROOT = "Data"
COUGH_FOLDER = "Cough"
NON_COUGH_FOLDER = "Noncough"
MODEL_PATH = "Models/cough_model.joblib"

if not os.path.exists("Models"):
    os.makedirs("Models")

# --- 2. FEATURE EXTRACTION ---
def extract_features(path):
    """Extracts features for a single audio file or segment"""
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
        if len(y) < 1000: return None

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        rms = librosa.feature.rms(y=y).mean()
        return np.concatenate([mfcc, [zcr, rms]])
    except Exception as e:
        return None

def extract_segment_features(segment, sr):
    """Helper for the sliding window counting logic"""
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13).mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(segment).mean()
    rms = librosa.feature.rms(y=segment).mean()
    return np.concatenate([mfcc, [zcr, rms]])

# --- 3. TRAINING LOGIC ---
def train_model():
    X, y_labels = [], []
    for label_name, label_value in [(COUGH_FOLDER, 1), (NON_COUGH_FOLDER, 0)]:
        folder_path = os.path.join(DATASET_ROOT, label_name)
        if not os.path.exists(folder_path): continue

        print(f"📂 Scanning: {label_name}")
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".wav", ".mp3")):
                feats = extract_features(os.path.join(folder_path, fname))
                if feats is not None:
                    X.append(feats)
                    y_labels.append(label_value)

    if len(X) > 0:
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(np.array(X), np.array(y_labels))
        joblib.dump(clf, MODEL_PATH)
        print(f"✅ Model trained and saved to {MODEL_PATH}")
    else:
        print("❌ No data found to train on.")

# --- 4. ACCURATE COUNTING & ANALYSIS (For app.py) ---
def analyze_audio(path, win_sec=0.5, hop_sec=0.25, prob_thresh=0.7):
    """
    Returns: (Label, Confidence, Cough_Count)
    Uses a sliding window to detect multiple coughs in one file.
    """
    if not os.path.exists(MODEL_PATH):
        return "Model Missing", 0.0, 0

    clf = joblib.load(MODEL_PATH)
    
    # Check the whole file first
    full_feats = extract_features(path)
    if full_feats is None: return "Error", 0.0, 0
    
    full_feats = full_feats.reshape(1, -1)
    probs = clf.predict_proba(full_feats)[0]
    pred = clf.predict(full_feats)[0]

    # If the whole file isn't a cough, don't bother counting
    if pred == 0:
        return "Non-Cough", float(probs[0]), 0

    # Sliding Window for Counting
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    win_len = int(win_sec * sr)
    hop_len = int(hop_sec * sr)
    cough_count = 0

    for start in range(0, len(y) - win_len, hop_len):
        segment = y[start:start + win_len]
        feats = extract_segment_features(segment, sr).reshape(1, -1)
        
        seg_probs = clf.predict_proba(feats)[0]
        seg_pred = clf.predict(feats)[0]

        if seg_pred == 1 and seg_probs[1] >= prob_thresh:
            cough_count += 1

    return "Cough", float(probs[1]), cough_count

# --- 5. TERMINAL EXECUTION (CLEANED) ---
if __name__ == "__main__":
    print("🚀 Cough Module: Training Mode")
    
    # Check if data exists before training
    if os.path.exists(os.path.join(DATASET_ROOT, COUGH_FOLDER)):
        train_model()
        print("✅ Training complete. Model is ready for app.py")
    else:
        print("❌ Data folder not found. Skipping training.")
    
    