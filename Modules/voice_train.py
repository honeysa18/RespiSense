import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

class VoiceClassifierTrainer:
    
    def extract_features(self, audio_file):
        """Extract acoustic features for ML"""
        audio, sr = librosa.load(audio_file, sr=22050)
        
        # Extract multiple features
        features = []
        
        # Jitter approximation (F0 variation)
        f0 = librosa.yin(audio, fmin=80, fmax=400, sr=sr)
        f0_valid = f0[f0 > 0]
        jitter = np.std(f0_valid) / np.mean(f0_valid) if len(f0_valid) > 0 else 0
        features.append(jitter)
        
        # Shimmer approximation (RMS variation)
        rms = librosa.feature.rms(y=audio)[0]
        shimmer = np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 0
        features.append(shimmer)
        
        # Additional features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        zero_crossing = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        features.extend([spectral_centroid, spectral_rolloff, zero_crossing])
        
        return np.array(features)
    
    
    def train_model(self, normal_folder, pathological_folder):
        """Train classifier on audio samples"""
        X, y = [], []
        
        print("🎤 Processing normal voice samples...")
        for file in os.listdir(normal_folder):
            if file.endswith(('.wav', '.mp3')):
                try:
                    features = self.extract_features(os.path.join(normal_folder, file))
                    X.append(features)
                    y.append(0)  # Normal
                except:
                    pass
        
        print("🎤 Processing pathological voice samples...")
        for file in os.listdir(pathological_folder):
            if file.endswith(('.wav', '.mp3')):
                try:
                    features = self.extract_features(os.path.join(pathological_folder, file))
                    X.append(features)
                    y.append(1)  # Pathological
                except:
                    pass
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n📊 Dataset: {len(X)} samples ({np.sum(y==0)} normal, {np.sum(y==1)} pathological)")
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        accuracy = clf.score(X_test, y_test)
        print(f"✅ Model trained. Test accuracy: {accuracy*100:.1f}%")
        
        # Save model
        with open('voice_classifier.pkl', 'wb') as f:
            pickle.dump(clf, f)
        print("💾 Model saved as 'voice_classifier.pkl'")
        
        return clf


if __name__ == "__main__":
    trainer = VoiceClassifierTrainer()
    
    normal_folder = input("Path to normal voice folder: ")
    pathological_folder = input("Path to pathological voice folder: ")
    
    trainer.train_model(normal_folder, pathological_folder)
