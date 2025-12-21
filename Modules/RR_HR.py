import pandas as pd
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

def process_vitals(df):
    """
    The function that app.py is looking for.
    It takes the data and returns the results.
    """
    try:
        # 1. Find the correct column
        if 'Linear Acceleration z (m/s^2)' in df.columns:
            signal = df['Linear Acceleration z (m/s^2)'].values 
        elif 'z' in df.columns: 
            signal = df['z'].values
        else:
            # If no column found, return zeros so the app doesn't crash
            return 0, 0, np.zeros(100), np.zeros(100), np.linspace(0, 10, 100)

        t = df['Time (s)'].values
        fs = 1 / np.median(np.diff(t)) 

        # --- Filter Helper ---
        def robust_bandpass_filter(data, lowcut, highcut, fs, order=4):
            nyq = 0.5 * fs
            low, high = lowcut / nyq, highcut / nyq
            if low <= 0 or high >= 1: return np.zeros_like(data)
            sos = butter(order, [low, high], btype='band', output='sos')
            return sosfiltfilt(sos, data)

        # 2. Process Breathing (RR)
        rr_signal = robust_bandpass_filter(signal, 0.1, 0.5, fs)
        rr_range = np.percentile(rr_signal, 95) - np.percentile(rr_signal, 5)
        rr_prom = 0.1 * rr_range if rr_range > 0 else 0.01
        rr_peaks, _ = find_peaks(rr_signal, distance=fs*2, prominence=rr_prom)
        rr_bpm = len(rr_peaks) / ((t[-1] - t[0]) / 60)

        # 3. Process Heart Rate (HR)
        hr_signal = robust_bandpass_filter(signal, 0.8, 2.5, fs)
        hr_range = np.percentile(hr_signal, 95) - np.percentile(hr_signal, 5)
        hr_prom = 0.05 * hr_range if hr_range > 0 else 0.005
        hr_peaks, _ = find_peaks(hr_signal, distance=fs*0.4, prominence=hr_prom)
        hr_bpm = len(hr_peaks) / ((t[-1] - t[0]) / 60)

        return rr_bpm, hr_bpm, rr_signal, hr_signal, t

    except Exception as e:
        print(f"Error in RR_HR logic: {e}")
        return 0, 0, np.zeros(100), np.zeros(100), np.linspace(0, 10, 100)