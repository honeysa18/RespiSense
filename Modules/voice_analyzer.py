import numpy as np
import librosa
import pickle
from .voice_biomarker import VoiceBiomarkerAnalyzer

class RespiSenseVoiceAnalyzer:
    """
    Unified voice analysis for RespiSense AI
    Combines acoustic biomarkers + ML classification
    """
    
    def __init__(self, classifier_path='voice_classifier.pkl'):
        # Load rule-based analyzer
        self.biomarker_analyzer = VoiceBiomarkerAnalyzer()
        
        # Load trained ML classifier
        try:
            with open(classifier_path, 'rb') as f:
                self.ml_classifier = pickle.load(f)
            print("✅ ML classifier loaded")
        except:
            print("⚠️ ML classifier not found, using biomarkers only")
            self.ml_classifier = None
    
    
    def extract_ml_features(self, audio_file):
        """Extract features for ML classifier (must match training)"""
        audio, sr = librosa.load(audio_file, sr=22050)
        
        features = []
        
        # Jitter
        f0 = librosa.yin(audio, fmin=80, fmax=400, sr=sr)
        f0_valid = f0[f0 > 0]
        jitter = np.std(f0_valid) / np.mean(f0_valid) if len(f0_valid) > 0 else 0
        features.append(jitter)
        
        # Shimmer
        rms = librosa.feature.rms(y=audio)[0]
        shimmer = np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 0
        features.append(shimmer)
        
        # Additional features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        zero_crossing = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        features.extend([spectral_centroid, spectral_rolloff, zero_crossing])
        
        return np.array(features).reshape(1, -1)
    
    
    def analyze_comprehensive(self, audio_file):
        """
        Complete voice analysis with both methods
        Returns structured output for Gemini fusion
        """
        print(f"\n🎤 RespiSense Voice Analysis: {audio_file}")
        print("=" * 60)
        
        # 1. Rule-based biomarker analysis
        biomarker_result = self.biomarker_analyzer.analyze(audio_file)
        
        if biomarker_result is None:
            return None
        
        # 2. ML classification (if available)
        ml_prediction = None
        ml_confidence = None
        
        if self.ml_classifier is not None:
            try:
                features = self.extract_ml_features(audio_file)
                ml_prediction = self.ml_classifier.predict(features)[0]
                ml_proba = self.ml_classifier.predict_proba(features)[0]
                ml_confidence = np.max(ml_proba) * 100
                
                status = "PATHOLOGICAL" if ml_prediction == 1 else "HEALTHY"
                print(f"\n🤖 ML Classification: {status} (confidence: {ml_confidence:.1f}%)")
            except Exception as e:
                print(f"⚠️ ML classification failed: {e}")
        
        # 3. Unified assessment
        result = {
            'audio_file': audio_file,
            
            # Biomarker metrics
            'jitter': biomarker_result['jitter'],
            'shimmer': biomarker_result['shimmer'],
            'biomarker_risk_level': biomarker_result['risk_level'],
            'biomarker_risk_score': biomarker_result['risk_score'],
            
            # ML prediction
            'ml_prediction': 'pathological' if ml_prediction == 1 else 'healthy',
            'ml_confidence': ml_confidence,
            
            # Combined assessment
            'findings': biomarker_result['findings'],
            'recommendation': biomarker_result['recommendation'],
            
            # For Gemini fusion layer
            'inflammation_detected': biomarker_result['risk_score'] >= 3 or ml_prediction == 1,
            'severity': self._calculate_severity(biomarker_result, ml_prediction)
        }
        
        print(f"\n📊 Final Assessment:")
        print(f"   Inflammation Detected: {'YES' if result['inflammation_detected'] else 'NO'}")
        print(f"   Severity: {result['severity']}")
        print("=" * 60)
        
        return result
    
    
    def _calculate_severity(self, biomarker_result, ml_prediction):
        """Combine both signals for severity estimate"""
        score = biomarker_result['risk_score']
        
        # Boost score if ML agrees
        if ml_prediction == 1:
            score += 1
        
        if score >= 5:
            return "HIGH"
        elif score >= 3:
            return "MODERATE"
        elif score >= 1:
            return "LOW"
        else:
            return "NORMAL"


# --- TESTING ---
if __name__ == "__main__":
    analyzer = RespiSenseVoiceAnalyzer()
    
    audio_path = input("Enter voice recording path: ")
    result = analyzer.analyze_comprehensive(audio_path)
    
    if result:
        print("\n✅ Analysis complete. Ready for Gemini fusion.")
        print(f"   Data for AI: {result}")
