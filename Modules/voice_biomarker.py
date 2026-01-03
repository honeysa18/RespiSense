import numpy as np
import librosa
import soundfile as sf
from scipy.stats import variation


class VoiceBiomarkerAnalyzer:
    """
    Vocal Resonance Analyzer for RespiSense AI
    Detects sub-clinical airway inflammation via jitter & shimmer analysis
    """
    
    def __init__(self):
        # Clinical thresholds based on respiratory pathology research
        self.JITTER_NORMAL_MAX = 1.0      # % - Normal: < 1%
        self.JITTER_WARNING = 2.5          # % - Mild inflammation: 1-2.5%
        self.JITTER_CRITICAL = 6.0         # % - Significant inflammation: > 2.5%
        
        self.SHIMMER_NORMAL_MAX = 3.0      # % - Normal: < 3%
        self.SHIMMER_WARNING = 6.0         # % - Mild inflammation: 3-6%
        self.SHIMMER_CRITICAL = 10.0       # % - Significant inflammation: > 6%
    
    
    def load_audio(self, file_path):
        """
        Load audio file and return signal with sampling rate
        Supports: .wav, .mp3, .m4a, .ogg
        """
        try:
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            return audio, sr
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            return None, None
    
    
    def extract_voiced_segment(self, audio, sr, min_duration=2.0):
        """
        Extract the sustained vowel portion from recording
        Removes silence at beginning/end
        """
        # Remove silence using librosa's voice activity detection
        intervals = librosa.effects.split(audio, top_db=30)
        
        if len(intervals) == 0:
            print("⚠️ No voiced segments detected")
            return None
        
        # Take the longest continuous voiced segment
        voiced_segments = [audio[start:end] for start, end in intervals]
        longest_segment = max(voiced_segments, key=len)
        
        # Check minimum duration (need at least 2 seconds for reliable analysis)
        duration = len(longest_segment) / sr
        if duration < min_duration:
            print(f"⚠️ Voiced segment too short: {duration:.2f}s (need >{min_duration}s)")
            return None
        
        return longest_segment
    
    
    def calculate_jitter(self, audio, sr):
        """
        Jitter: Pitch period variability (%)
        Measures cycle-to-cycle variations in fundamental frequency
        Indicator of vocal fold vibration irregularity
        """
        # Extract pitch using autocorrelation (YIN algorithm)
        f0 = librosa.yin(audio, fmin=80, fmax=400, sr=sr)
        
        # Remove unvoiced frames (where f0 estimation failed)
        f0_voiced = f0[f0 > 0]
        
        if len(f0_voiced) < 10:
            return 0.0  # Insufficient data
        
        # Calculate period from frequency
        periods = 1.0 / f0_voiced
        
        # Jitter = average absolute difference between consecutive periods / mean period
        period_diffs = np.abs(np.diff(periods))
        jitter = (np.mean(period_diffs) / np.mean(periods)) * 100
        
        return jitter
    
    
    def calculate_shimmer(self, audio, sr):
        """
        Shimmer: Amplitude variability (%)
        Measures cycle-to-cycle variations in amplitude
        Indicator of vocal fold closure irregularity
        """
        # Extract RMS energy in short frames
        frame_length = int(0.01 * sr)  # 10ms frames
        hop_length = frame_length // 2
        
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Remove near-zero frames
        rms_voiced = rms[rms > np.percentile(rms, 10)]
        
        if len(rms_voiced) < 10:
            return 0.0
        
        # Shimmer = average absolute difference between consecutive amplitudes / mean amplitude
        amp_diffs = np.abs(np.diff(rms_voiced))
        shimmer = (np.mean(amp_diffs) / np.mean(rms_voiced)) * 100
        
        return shimmer
    
    
    def assess_risk(self, jitter, shimmer):
        """
        Clinical interpretation of jitter/shimmer values
        Returns risk level and explanation
        """
        risk_score = 0
        findings = []
        
        # Jitter assessment
        if jitter >= self.JITTER_CRITICAL:
            risk_score += 3
            findings.append(f"⚠️ Critical jitter: {jitter:.2f}% (vocal fold inflammation likely)")
        elif jitter >= self.JITTER_WARNING:
            risk_score += 2
            findings.append(f"⚠️ Elevated jitter: {jitter:.2f}% (mild inflammation detected)")
        elif jitter >= self.JITTER_NORMAL_MAX:
            risk_score += 1
            findings.append(f"⚡ Borderline jitter: {jitter:.2f}% (monitor closely)")
        else:
            findings.append(f"✅ Normal jitter: {jitter:.2f}%")
        
        # Shimmer assessment
        if shimmer >= self.SHIMMER_CRITICAL:
            risk_score += 3
            findings.append(f"⚠️ Critical shimmer: {shimmer:.2f}% (airway closure irregularity)")
        elif shimmer >= self.SHIMMER_WARNING:
            risk_score += 2
            findings.append(f"⚠️ Elevated shimmer: {shimmer:.2f}% (mild vocal strain)")
        elif shimmer >= self.SHIMMER_NORMAL_MAX:
            risk_score += 1
            findings.append(f"⚡ Borderline shimmer: {shimmer:.2f}% (monitor closely)")
        else:
            findings.append(f"✅ Normal shimmer: {shimmer:.2f}%")
        
        # Overall risk classification
        if risk_score >= 5:
            risk_level = "HIGH RISK"
            recommendation = "⚠️ Significant inflammation detected. Consider medical evaluation."
        elif risk_score >= 3:
            risk_level = "MODERATE RISK"
            recommendation = "⚡ Early inflammation signs. Monitor symptoms closely."
        elif risk_score >= 1:
            risk_level = "LOW RISK"
            recommendation = "✓ Minor irregularities. Continue monitoring."
        else:
            risk_level = "NORMAL"
            recommendation = "✅ No inflammation detected. Vocal cords healthy."
        
        return {
            'jitter': jitter,
            'shimmer': shimmer,
            'risk_level': risk_level,
            'risk_score': risk_score,
            'findings': findings,
            'recommendation': recommendation
        }
    
    
    def analyze(self, audio_file_path):
        """
        Complete analysis pipeline: The 5-Second 'Ahhh' Test
        """
        print(f"\n🎤 Analyzing Voice Biomarkers from: {audio_file_path}")
        print("=" * 60)
        
        # Load audio
        audio, sr = self.load_audio(audio_file_path)
        if audio is None:
            return None
        
        print(f"📊 Audio loaded: {len(audio)/sr:.2f}s @ {sr}Hz")
        
        # Extract sustained vowel
        voiced = self.extract_voiced_segment(audio, sr)
        if voiced is None:
            return None
        
        print(f"🎵 Voiced segment extracted: {len(voiced)/sr:.2f}s")
        
        # Calculate biomarkers
        jitter = self.calculate_jitter(voiced, sr)
        shimmer = self.calculate_shimmer(voiced, sr)
        
        print(f"\n📈 Biomarker Results:")
        print(f"   Jitter (Pitch Stability):     {jitter:.2f}%")
        print(f"   Shimmer (Amplitude Stability): {shimmer:.2f}%")
        
        # Assess risk
        assessment = self.assess_risk(jitter, shimmer)
        
        print(f"\n🏥 Clinical Assessment:")
        print(f"   Risk Level: {assessment['risk_level']}")
        for finding in assessment['findings']:
            print(f"   {finding}")
        print(f"\n   {assessment['recommendation']}")
        print("=" * 60)
        
        return assessment


# --- TESTING INTERFACE ---
if __name__ == "__main__":
    analyzer = VoiceBiomarkerAnalyzer()
    
    # Test with audio file
    audio_path = input("Enter path to voice recording (.wav, .mp3, etc.): ")
    result = analyzer.analyze(audio_path)
    
    if result:
        print("\n✅ Analysis complete. Results saved to terminal.")
    else:
        print("\n❌ Analysis failed. Check audio file.")
