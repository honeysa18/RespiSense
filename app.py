import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

# Import custom modules
# Note: Ensure these files are in a folder named 'Modules' with an __init__.py file
from Modules import RR_HR, cough, aqi

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="RespiSense AI", page_icon="🫁", layout="wide")

# --- 1. LOAD AI MODEL ---
@st.cache_resource
def load_ai_brain():
    model_path = 'Models/respi_model.h5'
    if os.path.exists(model_path):
        return load_model(model_path)
    return None

model = load_ai_brain()

# --- 2. AI DIAGNOSIS HELPER ---
def get_ai_prediction(signal, fs):
    f, t_spec, Sxx = spectrogram(signal, fs, nperseg=256, noverlap=128)
    S_db = 10 * np.log10(Sxx + 1e-10)
    S_min, S_max = S_db.min(), S_db.max()
    img = 255 * (S_db - S_min) / (S_max - S_min)
    img = cv2.resize(img.astype(np.uint8), (224, 224))
    img_rgb = np.stack((img,)*3, axis=-1)
    img_batch = np.expand_dims(img_rgb, axis=0)
    score = model.predict(img_batch, verbose=0)[0][0]
    return score

# --- 3. MAIN UI HEADER ---
st.title("🫁 RespiSense: Integrated Respiratory Intelligence")
st.markdown("### *Multi-Modal Monitoring: Kinematic, Acoustic & Environmental Correlation*")

# Initialize session state for risk assessment persistence
if 'rr' not in st.session_state: st.session_state.rr = 0
if 'risk_score' not in st.session_state: st.session_state.risk_score = 0
if 'cough_count' not in st.session_state: st.session_state.cough_count = 0
if 'current_aqi' not in st.session_state: st.session_state.current_aqi = 0

# --- 4. SIDEBAR: ENVIRONMENTAL CONTEXT ---
with st.sidebar:
    st.header("🌍 Environmental Data")
    selected_city = st.selectbox("Select Patient City", ["Chennai", "Delhi", "Bangalore"])
    user_api_key = st.text_input("OpenWeather API Key (Optional)", type="password")

    if st.button("Check Air Quality"):
        env_data = aqi.get_air_quality(selected_city, user_api_key)
        st.session_state.current_aqi = env_data['aqi']
        st.session_state.current_status = env_data['status']
        st.session_state.current_pm25 = env_data['pm25']
        
    st.metric("AQI Level", f"{st.session_state.current_aqi}/5", st.session_state.get('current_status', "Not Checked"))

# --- 5. TABS FOR ANALYSIS ---
tab_vitals, tab_acoustic = st.tabs(["🛌 Passive Check (Vitals)", "🎤 Active Mode (Cough)"])

# ==========================================
# TAB 1: PASSIVE CHECK (CSV SENSOR DATA)
# ==========================================
with tab_vitals:
    st.header("Chest Motion Analysis")
    uploaded_csv = st.file_uploader("Upload Phyphox CSV", type=['csv'], key="csv_up")
    
    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        with st.spinner("Analyzing Vitals..."):
            rr, hr, rr_wave, hr_wave, t = RR_HR.process_vitals(df)
            st.session_state.rr = rr
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Respiratory Rate", f"{rr:.1f} BPM", delta="Normal" if 12<=rr<=20 else "Abnormal")
        m2.metric("Heart Rate", f"{hr:.1f} BPM")
        
        if model is not None:
            sig_col = 'Linear Acceleration z (m/s^2)' if 'Linear Acceleration z (m/s^2)' in df.columns else 'z'
            raw_sig = df[sig_col].values
            fs = 1 / np.median(np.diff(t))
            
            chunk_size = int(10 * fs)
            if len(raw_sig) > chunk_size:
                start = (len(raw_sig) // 2) - (chunk_size // 2)
                chunk = raw_sig[start : start + chunk_size]
                risk_score = get_ai_prediction(chunk, fs)
                st.session_state.risk_score = risk_score
                
                # Fixed Fused Logic
                vitals_abnormal = rr < 12 or rr > 25
                if vitals_abnormal or risk_score > 0.4:
                    diagnosis, status_color = "⚠️ Abnormal Pattern", "inverse"
                else:
                    diagnosis, status_color = "✅ Healthy Pattern", "normal"
                
                m3.metric("AI Diagnosis", diagnosis, f"Risk: {risk_score:.0%}", delta_color=status_color)
            else:
                st.warning("Data too short for AI analysis.")

        # Clinical Visualization
        st.divider()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        plt.subplots_adjust(hspace=0.4)
        
        ax1.plot(t, rr_wave, color='#1f77b4', label='Respiration Wave')
        ax1.set_title("Respiratory Waveform (Filtered 0.1-0.5Hz)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        ax2.plot(t, hr_wave, color='#d62728', label='Cardiac Signal')
        ax2.set_title("Seismocardiogram (Heart Micro-vibrations)")
        ax2.set_xlabel("Time (s)")
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        st.pyplot(fig)


# ==========================================
# TAB 2: ACTIVE MODE (AUDIO COUGH DATA)
# ==========================================
with tab_acoustic:
    st.header("Acoustic Symptom Tracking")
    uploaded_audio = st.file_uploader("Upload Audio Clip", type=['wav', 'mp3'], key="audio_up")
    
    if uploaded_audio:
        temp_path = "temp_audio_file.wav"
        with open(temp_path, "wb") as f:
            f.write(uploaded_audio.getbuffer())
        
        label, confidence, count = cough.analyze_audio(temp_path)
        st.session_state.cough_count = count
        
        a1, a2 = st.columns(2)
        a1.metric("Detected Coughs", count)
        a2.metric("AI Confidence", f"{confidence:.1%}")
        
        if count > 5: st.error(f"High Irritation Alert")
        elif count > 0: st.warning(f"Mild Irritation")
        
        if os.path.exists(temp_path): os.remove(temp_path)

# ==========================================
# MASTER RISK CORRELATION
# ==========================================
st.divider()
st.header("📋 Master Health Summary")

# Score Calculation
score_rr = 35 if (st.session_state.rr > 22 or (0 < st.session_state.rr < 10)) else 0
score_ai = 25 if st.session_state.risk_score > 0.5 else 0
score_cough = 20 if st.session_state.cough_count > 5 else 0
score_env = 20 if st.session_state.current_aqi >= 4 else 0

total_score = score_rr + score_ai + score_cough + score_env

col_left, col_right = st.columns([1, 2])
with col_left:
    st.metric("Total Risk Index", f"{total_score}%")
    st.progress(total_score / 100)
with col_right:
    if total_score >= 70: st.error("🚨 CRITICAL RISK: Immediate medical review recommended.")
    elif total_score >= 40: st.warning("⚠️ MODERATE RISK: Respiratory symptoms and triggers detected.")
    else: st.success("✅ STABLE: No high-risk correlations found.")