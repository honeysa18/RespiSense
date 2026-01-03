import wfdb
import numpy as np
import soundfile as sf
import os
from pathlib import Path

class VoicedDatasetConverter:
    """
    Converts PhysioNet VOICED database (.dat/.hea) to .wav files
    """
    
    def __init__(self, voiced_db_path, output_path="Audio_Samples"):
        self.voiced_path = Path(voiced_db_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Create output folders
        (self.output_path / "healthy").mkdir(exist_ok=True)
        (self.output_path / "pathological").mkdir(exist_ok=True)
    
    
    def convert_record(self, record_name):
        """
        Convert single .dat/.hea record to .wav
        """
        try:
            # Read WFDB record
            record = wfdb.rdrecord(str(self.voiced_path / record_name))
            
            # Extract audio signal (first channel)
            audio = record.p_signal[:, 0]
            
            # Get sampling frequency
            fs = record.fs
            
            # Normalize audio to [-1, 1] range
            audio = audio / np.max(np.abs(audio))
            
            return audio, fs
            
        except Exception as e:
            print(f"❌ Error reading {record_name}: {e}")
            return None, None
    
    
    def get_label_from_filename(self, filename):
        """
        VOICED database naming convention:
        - Healthy: files typically contain 'n' or are in healthy range
        - Pathological: majority of files
        
        You may need to check RECORDS file for exact labeling
        """
        # This is a placeholder - check your RECORDS file for actual labels
        # VOICED typically numbers healthy as 001-058, pathological as 059-208
        try:
            num = int(''.join(filter(str.isdigit, filename)))
            return "healthy" if num <= 58 else "pathological"
        except:
            return "unknown"
    
    
    def convert_all(self):
        """
        Convert entire VOICED database
        """
        print("🎤 Starting VOICED Database Conversion...")
        print("=" * 60)
        
        # Look for all .hea files
        hea_files = list(self.voiced_path.glob("*.hea"))
        
        if len(hea_files) == 0:
            print("❌ No .hea files found. Check the path.")
            return
        
        print(f"📂 Found {len(hea_files)} records")
        
        converted_count = {"healthy": 0, "pathological": 0, "failed": 0}
        
        for hea_file in hea_files:
            record_name = hea_file.stem  # Filename without extension
            
            # Convert to audio
            audio, fs = self.convert_record(record_name)
            
            if audio is not None:
                # Determine label
                label = self.get_label_from_filename(record_name)
                
                if label in ["healthy", "pathological"]:
                    # Save as WAV
                    output_file = self.output_path / label / f"{record_name}.wav"
                    sf.write(output_file, audio, fs)
                    converted_count[label] += 1
                    print(f"✅ {record_name} → {label}/{record_name}.wav")
                else:
                    converted_count["failed"] += 1
            else:
                converted_count["failed"] += 1
        
        print("=" * 60)
        print(f"🎉 Conversion Complete!")
        print(f"   Healthy: {converted_count['healthy']} files")
        print(f"   Pathological: {converted_count['pathological']} files")
        print(f"   Failed: {converted_count['failed']} files")
        print(f"\n📁 Output folder: {self.output_path.absolute()}")


# --- USAGE ---
if __name__ == "__main__":
    print("VOICED Database Converter")
    print("=" * 60)
    
    db_path = input("Enter path to VOICED database folder (with .dat/.hea files): ")
    
    converter = VoicedDatasetConverter(db_path)
    converter.convert_all()
