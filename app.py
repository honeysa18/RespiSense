import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

# Import your custom modules
from Modules import RR_HR, cough, aqi


# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="RespiSense AI", page_icon="🫁", layout="wide")

# --- 1. LOAD AI MODEL ---
@st.cache_resource
def load_ai_brain():
    if os.path.exists('Models/respi_model.h5'):
        return load_model('Models/respi_model.h5')
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

# --- 4. SIDEBAR: ENVIRONMENTAL CONTEXT ---
with st.sidebar:
    st.header("🌍 Environmental Data")
    st.info("Retrieve live air quality data to assess external triggers.")
    
    selected_city = st.selectbox("Select Patient City", ["Chennai", "Delhi", "Bangalore"])
    user_api_key = st.text_input("OpenWeather API Key (Optional)", type="password")
    
    # Initialize session state for AQI
    if 'current_aqi' not in st.session_state:
        st.session_state['current_aqi'] = 0
        st.session_state['current_status'] = "Not Checked"

    if st.button("Check Air Quality"):
        env_data = aqi.get_air_quality(selected_city, user_api_key)
        st.session_state['current_aqi'] = env_data['aqi']
        st.session_state['current_status'] = env_data['status']
        st.session_state['current_pm25'] = env_data['pm25']
        
    st.metric("AQI Level", f"{st.session_state['current_aqi']}/5", st.session_state['current_status'])
    if 'current_pm25' in st.session_state:
        st.caption(f"PM2.5: {st.session_state['current_pm25']} µg/m³")

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
        
        # Process Vitals via RR_HR.py
        with st.spinner("Analyzing Vitals..."):
            rr, hr, rr_wave, hr_wave, t = RR_HR.process_vitals(df)
        
        # Display Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Respiratory Rate", f"{rr:.1f} BPM", delta="Normal" if 12<=rr<=20 else "Abnormal", delta_color="normal" if 12<=rr<=20 else "inverse")
        m2.metric("Heart Rate", f"{hr:.1f} BPM")
        
        # AI Pattern Recognition
        if model is not None:
            sig_col = 'Linear Acceleration z (m/s^2)' if 'Linear Acceleration z (m/s^2)' in df.columns else 'z'
            raw_sig = df[sig_col].values
            fs = 1 / np.median(np.diff(t))
            
            chunk_size = int(10 * fs)
            if len(raw_sig) > chunk_size:
                start = (len(raw_sig) // 2) - (chunk_size // 2)
                chunk = raw_sig[start : start + chunk_size]
                risk_score = get_ai_prediction(chunk, fs)
                
                diagnosis = "⚠️ Abnormal Pattern" if risk_score > 0.5 else "✅ Healthy Pattern"
                m3.metric("AI Diagnosis", diagnosis, f"Risk: {risk_score:.0%}")
            else:
                m3.warning("Data too short for AI")
        else:
            m3.warning("AI Model not loaded.")

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
        
        with st.spinner("Scanning for cough events..."):
            label, confidence, count = cough.analyze_audio(temp_path)
        
        a1, a2 = st.columns(2)
        a1.metric("Detected Coughs", count)
        a2.metric("AI Confidence", f"{confidence:.1%}")
        
        if count > 5:
            st.error(f"High Irritation Alert: {count} coughs detected.")
        elif count > 0:
            st.warning(f"Mild Irritation: {count} coughs detected.")
        else:
            st.success("Clear Sample: No coughing patterns found.")

        if os.path.exists(temp_path):
            os.remove(temp_path)

# ==========================================
# MASTER RISK CORRELATION (THE FINAL SCORE)
# ==========================================
st.divider()
st.header("📋 Master Health Summary & Risk Assessment")

# --- INITIALIZE DEFAULTS ---
# This ensures the correlation block always appears
res_rr = 0
res_ai = 0
res_cough = 0
res_env = 0

# 1. Check Vitals (from locals or session state)
if 'rr' in locals():
    res_rr = 35 if (rr > 22 or rr < 10) else 0

# 2. Check AI Prediction
if 'risk_score' in locals():
    res_ai = 25 if risk_score > 0.5 else 0

# 3. Check Cough Count (This often resets, so we check if 'count' exists)
if 'count' in locals():
    res_cough = 20 if count > 5 else 0

# 4. Check Environment (from session state)
current_aqi = st.session_state.get('current_aqi', 0)
res_env = 20 if current_aqi >= 4 else 0

# --- FINAL CALCULATION ---
total_score = res_rr + res_ai + res_cough + res_env

c_res, c_msg = st.columns([1, 2])

with c_res:
    st.metric("Total Risk Index", f"{total_score}%")
    st.progress(total_score / 100)

with c_msg:
    if total_score >= 70:
        st.error("🚨 CRITICAL RISK: High correlation between respiratory distress and environment.")
    elif total_score >= 40:
        st.warning("⚠️ MODERATE RISK: Respiratory symptoms and triggers detected.")
    elif total_score > 0:
        st.info("✅ MONITORING: Some data detected. Results will update as you upload more files.")
    else:
        st.success("✅ STABLE: No risks detected yet. Please upload data to begin.")

st.info("💡 Note: This is an AI-assisted prototype for K!Hacks and not for clinical diagnosis.")