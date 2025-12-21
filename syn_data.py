import pandas as pd
import numpy as np
import os

def generate_synthetic_files():
    # --- CONFIGURATION (Matching your Phyphox Data) ---
    FS = 202.0       # Sampling Rate (Hz)
    DURATION = 40.0  # Seconds
    
    # Create Time Axis
    num_samples = int(FS * DURATION)
    t = np.linspace(0, DURATION, num_samples)
    
    # --- HELPER: GENERATE BREATHING SIGNALS ---
    def get_breathing_wave(freq_hz, amplitude, pattern="normal"):
        # Base Noise (Sensor noise)
        noise = np.random.normal(0, 0.005, len(t))
        
        if pattern == "normal":
            # Simple Sine Wave + Heartbeat Ripple
            wave = amplitude * np.sin(2 * np.pi * freq_hz * t)
            heartbeat = (amplitude * 0.1) * np.sin(2 * np.pi * 1.2 * t)
            return wave + heartbeat + noise
            
        elif pattern == "cheyne_stokes":
            # Waxing and Waning Amplitude (Deep -> Shallow -> Deep)
            # Modulation cycle every 20 seconds
            envelope = 0.5 * (1 + np.sin(2 * np.pi * (1/20) * t)) 
            wave = envelope * (amplitude * np.sin(2 * np.pi * freq_hz * t))
            return wave + noise

    # --- 1. GENERATE TACHYPNEA (Rapid: 35 BPM) ---
    print("Generating Tachypnea (Rapid Breathing)...")
    z_tach = get_breathing_wave(freq_hz=0.6, amplitude=0.4, pattern="normal")
    save_csv("abnormal_tachypnea.csv", t, z_tach)

    # --- 2. GENERATE BRADYPNEA (Slow: 8 BPM) ---
    print("Generating Bradypnea (Slow Breathing)...")
    z_brad = get_breathing_wave(freq_hz=0.13, amplitude=0.2, pattern="normal")
    save_csv("abnormal_bradypnea.csv", t, z_brad)

    # --- 3. GENERATE IRREGULAR (Cheyne-Stokes) ---
    print("Generating Irregular Pattern...")
    z_irreg = get_breathing_wave(freq_hz=0.3, amplitude=0.3, pattern="cheyne_stokes")
    save_csv("abnormal_irregular.csv", t, z_irreg)

    print("\n✅ Success! 3 CSV files generated in your folder.")

def save_csv(filename, t, z_values):
    # Create dummy X, Y and Absolute values to match Phyphox format
    x = np.random.normal(0, 0.02, len(t))
    y = np.random.normal(0, 0.02, len(t))
    
    # Calculate Absolute (Approximate)
    abs_acc = np.sqrt(x**2 + y**2 + z_values**2)
    
    df = pd.DataFrame({
        'Time (s)': t,
        'Linear Acceleration x (m/s^2)': x,
        'Linear Acceleration y (m/s^2)': y,
        'Linear Acceleration z (m/s^2)': z_values,
        'Absolute acceleration (m/s^2)': abs_acc
    })
    
    df.to_csv(filename, index=False)
    print(f" -> Saved {filename}")

if __name__ == "__main__":
    generate_synthetic_files()