import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from google import genai
from chatbot import chatbot_response

# Import custom modules
from Modules import vitals, cough
from Modules.voice_analyzer import RespiSenseVoiceAnalyzer

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
st.title("🫁 RespiSense AI: Intelligent Respiratory Profiler")
st.markdown("### *Your Personal Bio-Weather Station | Multi-Modal Data Fusion with Gemini*")

# Initialize session state
if 'rr' not in st.session_state: st.session_state.rr = 0
if 'hr' not in st.session_state: st.session_state.hr = 0
if 'risk_score' not in st.session_state: st.session_state.risk_score = 0
if 'cough_count' not in st.session_state: st.session_state.cough_count = 0
if 'current_aqi' not in st.session_state: st.session_state.current_aqi = 1
if 'current_status' not in st.session_state: st.session_state.current_status = "Not Checked"
if 'current_pm25' not in st.session_state: st.session_state.current_pm25 = 0.0
if 'humidity' not in st.session_state: st.session_state.humidity = 50
if 'pollen_index' not in st.session_state: st.session_state.pollen_index = 0.0
if 'voice_data' not in st.session_state: st.session_state.voice_data = None
if 'show_env' not in st.session_state: st.session_state.show_env = False
if 'gemini_key' not in st.session_state: st.session_state.gemini_key = None

# --- 4. SIDEBAR: API KEYS & ENVIRONMENTAL CONTEXT ---
with st.sidebar:
    st.header("🧠 AI Configuration")
    gemini_api_key = st.text_input("Gemini API Key", type="password", help="Get from aistudio.google.com")
    
    if gemini_api_key:
        try:
            client = genai.Client(api_key=gemini_api_key)
            # Test connection
            test = client.models.generate_content(
                model='models/gemini-2.5-flash',
                contents='test'
            )
            st.session_state.gemini_key = gemini_api_key
            st.success("✅ Gemini Connected")
        except Exception as e:
            st.error(f"⚠️ API Key Error: {str(e)}")
    
    st.divider()
    st.header("🌍 Environmental Monitoring")
    
    # Embedded HTML with data bridge
    html_monitor = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8">
      <style>
        body { 
          font-family: 'Segoe UI', sans-serif; 
          padding: 15px; 
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border-radius: 10px;
        }
        #info { 
          line-height: 2; 
          font-size: 14px;
          background: rgba(255,255,255,0.1);
          padding: 15px;
          border-radius: 8px;
          margin-top: 10px;
        }
        .status { 
          font-weight: bold; 
          font-size: 13px;
          padding: 8px;
          background: rgba(0,0,0,0.2);
          border-radius: 5px;
          margin-bottom: 10px;
        }
        h3 { margin-top: 0; }
      </style>
    </head>
    <body>
    <h3>🌍 Live Environmental Scan</h3>
    <div id="status" class="status">📡 Acquiring geolocation...</div>
    <div id="info">Waiting for data...</div>
    
    <script>
    const API_KEY = "14f908d5b1ff835896a635a7d8adbded";
    const envData = { aqi: 1, pm25: 0, humidity: 50, pollen: 0, temp: 0 };
    
    if (!("geolocation" in navigator)) {
      document.getElementById("status").innerHTML = "⚠️ Geolocation not supported";
    } else {
      navigator.geolocation.getCurrentPosition(position => {
        const lat = position.coords.latitude;
        const lon = position.coords.longitude;
        document.getElementById("status").innerHTML = `📍 <strong>Location:</strong> ${lat.toFixed(3)}, ${lon.toFixed(3)}`;
        let output = "";
        
        // Weather API
        fetch(`https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&units=metric&appid=${API_KEY}`)
          .then(r => r.json())
          .then(w => {
            envData.humidity = w.main.humidity;
            envData.temp = w.main.temp;
            output += `🌡️ <strong>Temperature:</strong> ${w.main.temp.toFixed(1)}°C<br>`;
            output += `💧 <strong>Humidity:</strong> ${w.main.humidity}%<br>`;
            document.getElementById("info").innerHTML = output;
          })
          .catch(() => {
            output += `⚠️ Weather data unavailable<br>`;
            document.getElementById("info").innerHTML = output;
          });
        
        // Air Quality API
        fetch(`https://api.openweathermap.org/data/2.5/air_pollution?lat=${lat}&lon=${lon}&appid=${API_KEY}`)
          .then(r => r.json())
          .then(a => {
            const aqi = a.list[0].main.aqi;
            const pm25 = a.list[0].components.pm2_5;
            envData.aqi = aqi;
            envData.pm25 = pm25;
            
            const aqiColors = ["#00e400", "#9ACD32", "#ff7e00", "#ff0000", "#8f3f97"];
            const aqiLabels = ["Good", "Fair", "Moderate", "Poor", "Very Poor"];
            
            output += `<span style="color:${aqiColors[aqi-1]}">🌫️ <strong>AQI:</strong> ${aqi}/5 (${aqiLabels[aqi-1]})</span><br>`;
            output += `🔬 <strong>PM2.5:</strong> ${pm25.toFixed(1)} µg/m³<br>`;
            document.getElementById("info").innerHTML = output;
          })
          .catch(() => {
            output += `⚠️ Air quality data unavailable<br>`;
            document.getElementById("info").innerHTML = output;
          });
        
        // Pollen API
        fetch(`https://air-quality-api.open-meteo.com/v1/air-quality?latitude=${lat}&longitude=${lon}&hourly=grass_pollen`)
          .then(r => r.json())
          .then(p => {
            const pollen = p.hourly.grass_pollen[0];
            envData.pollen = pollen ?? 0;
            output += `🌾 <strong>Grass Pollen:</strong> ${pollen ? pollen.toFixed(1) : "N/A"}<br>`;
            document.getElementById("info").innerHTML = output;
          })
          .catch(() => {
            output += `⚠️ Pollen data unavailable<br>`;
            document.getElementById("info").innerHTML = output;
          });
      }, () => {
        document.getElementById("status").innerHTML = "❌ <strong>Location access denied</strong>";
        document.getElementById("info").innerHTML = "Enable location permissions to fetch environmental data.";
      });
    }
    </script>
    </body>
    </html>
    """
    # Read your working Environment.html file
if st.button("🌦️ Load Environmental Data", use_container_width=True):
    st.session_state.show_env = True

if st.session_state.show_env:
    # Load your working HTML file
    try:
        with open('Environment.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        st.components.v1.html(html_content, height=400, scrolling=True)
    except FileNotFoundError:
        st.error("⚠️ Environment.html file not found. Place it in the project root folder.")
    
    st.divider()
    st.caption("📝 Enter values from monitor above:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.current_aqi = st.selectbox(
            "AQI Level",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: ["Good", "Fair", "Moderate", "Poor", "Very Poor"][x-1],
            index=4,  # Default to 5 (Very Poor) based on your data
            key="aqi_select"
        )
        
        st.session_state.current_pm25 = st.number_input(
            "PM2.5 (µg/m³)",
            min_value=0.0,
            max_value=500.0,
            value=88.63,  # Your actual value
            step=0.1,
            key="pm25_input"
        )
    
    with col2:
        st.session_state.humidity = st.number_input(
            "Humidity (%)",
            min_value=0,
            max_value=100,
            value=81,  # Your actual value
            key="humidity_input"
        )
        
        st.session_state.pollen_index = st.number_input(
            "Grass Pollen",
            min_value=0.0,
            max_value=100.0,
            value=0.0,  # Not available in your case
            step=0.1,
            key="pollen_input"
        )
    
    # Update status
    aqi_status = ["Good", "Fair", "Moderate", "Poor", "Very Poor"]
    st.session_state.current_status = aqi_status[st.session_state.current_aqi - 1]
    
    st.success(f"✅ Environmental data: AQI {st.session_state.current_aqi}/5 ({st.session_state.current_status}), PM2.5: {st.session_state.current_pm25:.1f} µg/m³")

# --- 5. TABS FOR MULTI-MODAL ANALYSIS ---
tab_vitals, tab_acoustic, tab_voice = st.tabs([
    "🛌 Kinematic Monitor", 
    "🎤 Cough Tracking", 
    "🎵 Voice Biomarker"
])

# ==========================================
# TAB 1: KINEMATIC VITAL MONITOR
# ==========================================
with tab_vitals:
    st.header("Chest Motion Analysis (Seismocardiography)")
    uploaded_csv = st.file_uploader("📂 Upload Phyphox Accelerometer CSV", type=['csv'], key="csv_up")
    
    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        with st.spinner("🔬 Analyzing Vitals..."):
            rr, hr, rr_wave, hr_wave, t = vitals.process_vitals(df)
            st.session_state.rr = rr
            st.session_state.hr = hr
        
        m1, m2, m3 = st.columns(3)
        
        rr_status = "Normal" if 12 <= rr <= 20 else "Abnormal"
        rr_delta_color = "normal" if rr_status == "Normal" else "inverse"
        m1.metric("Respiratory Rate", f"{rr:.1f} BPM", delta=rr_status, delta_color=rr_delta_color)
        
        hr_status = "Normal" if 60 <= hr <= 100 else "Check"
        m2.metric("Heart Rate", f"{hr:.1f} BPM", delta=hr_status)
        
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
                
                vitals_abnormal = rr < 12 or rr > 25
                if vitals_abnormal or risk_score > 0.4:
                    diagnosis, status_color = "⚠️ Abnormal", "inverse"
                else:
                    diagnosis, status_color = "✅ Normal", "normal"
                
                m3.metric("AI Breathing Pattern", diagnosis, f"Risk: {risk_score:.0%}", delta_color=status_color)
            else:
                st.warning("⚠️ Data too short for AI analysis.")

        # Waveform Visualization
        st.divider()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        plt.subplots_adjust(hspace=0.4)
        
        ax1.plot(t, rr_wave, color='#1f77b4', linewidth=1.5)
        ax1.set_title("Respiratory Waveform (0.1-0.5 Hz Filtered)", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Amplitude (m/s²)")
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        ax2.plot(t, hr_wave, color='#d62728', linewidth=1.5)
        ax2.set_title("Seismocardiogram (Cardiac Micro-vibrations)", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude (m/s²)")
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        st.pyplot(fig)
        plt.close()

# ==========================================
# TAB 2: ACOUSTIC COUGH TRACKING
# ==========================================
with tab_acoustic:
    st.header("Acoustic Biomarker Engine")
    uploaded_audio = st.file_uploader("📂 Upload Audio Recording", type=['wav', 'mp3'], key="audio_up")
    
    if uploaded_audio:
        temp_path = "temp_cough_audio.wav"
        with open(temp_path, "wb") as f:
            f.write(uploaded_audio.getbuffer())
        
        with st.spinner("🎧 Analyzing cough patterns..."):
            label, confidence, count = cough.analyze_audio(temp_path)
            st.session_state.cough_count = count
        
        a1, a2, a3 = st.columns(3)
        a1.metric("Detected Coughs", count)
        a2.metric("AI Confidence", f"{confidence:.1%}")
        
        if count > 5:
            a3.error("🚨 High Irritation")
        elif count > 0:
            a3.warning("⚠️ Mild Irritation")
        else:
            a3.success("✅ No Coughs")
        
        if os.path.exists(temp_path): 
            os.remove(temp_path)

# ==========================================
# TAB 3: VOCAL RESONANCE ANALYZER
# ==========================================
with tab_voice:
    st.header("Vocal Biomarker Analysis")
    st.caption("📝 Record a sustained 'Ahhh' sound for 3-5 seconds")
    
    uploaded_voice = st.file_uploader("📂 Upload Voice Recording", type=['wav', 'mp3', 'm4a'], key="voice_up")
    
    if uploaded_voice:
        temp_voice = "temp_voice_file.wav"
        with open(temp_voice, "wb") as f:
            f.write(uploaded_voice.getbuffer())
        
        with st.spinner("🔊 Analyzing vocal biomarkers..."):
            try:
                analyzer = RespiSenseVoiceAnalyzer()
                result = analyzer.analyze_comprehensive(temp_voice)
                
                if result:
                    st.session_state.voice_data = result
                    
                    v1, v2, v3, v4 = st.columns(4)
                    v1.metric("Jitter", f"{result['jitter']:.2f}%")
                    v2.metric("Shimmer", f"{result['shimmer']:.2f}%")
                    v3.metric("ML Classification", result['ml_prediction'].upper())
                    
                    if result['inflammation_detected']:
                        v4.error(f"⚠️ {result['severity']}")
                    else:
                        v4.success("✅ Normal")
                    
                    with st.expander("📋 Detailed Voice Analysis"):
                        for finding in result['findings']:
                            st.write(finding)
                        st.info(result['recommendation'])
                else:
                    st.error("❌ Voice analysis failed. Check audio quality.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        if os.path.exists(temp_voice): 
            os.remove(temp_voice)

# ==========================================
# GEMINI CLINICAL FUSION LAYER
# ==========================================
st.divider()
st.header("🧠 Master Clinical Assessment (Gemini-Powered)")

def generate_gemini_clinical_report(patient_data):
    """
    RespiSense Layer 2: Agentic Data Fusion
    Identifies 'Invisible Trigger Intersections'
    """
    
    prompt = f"""You are RespiSense AI, an intelligent respiratory monitoring system that correlates INTERNAL PHYSIOLOGY with EXTERNAL ENVIRONMENT to detect invisible triggers before clinical severity.

**PATIENT DATA SNAPSHOT:**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 PHYSIOLOGICAL METRICS (Internal):
• Respiratory Rate: {patient_data['respiratory_rate']:.1f} breaths/min (Normal: 12-20)
• Heart Rate: {patient_data['heart_rate']:.1f} BPM (Normal: 60-100)
• AI Breathing Pattern Risk: {patient_data['ai_risk_score']:.1%}
• Pattern Classification: {"Abnormal/Pathological" if patient_data['ai_risk_score'] > 0.5 else "Normal/Healthy"}

🎤 ACOUSTIC BIOMARKERS:
• Cough Episodes: {patient_data['cough_count']} detected
• Voice Jitter: {patient_data.get('voice_jitter', 'Not tested')}
• Voice Shimmer: {patient_data.get('voice_shimmer', 'Not tested')}
• Vocal Inflammation: {patient_data.get('voice_inflammation', 'Not tested')}

🌍 ENVIRONMENTAL CONTEXT (External):
• Air Quality Index: {patient_data['aqi']}/5 ({patient_data['aqi_status']})
• PM2.5 Particulate Matter: {patient_data['pm25']} µg/m³
• Humidity: {patient_data['humidity']}%
• Grass Pollen Level: {patient_data['pollen']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**CLINICAL REASONING TASK:**
Perform multi-modal data fusion to generate an actionable clinical assessment:

1. **INVISIBLE TRIGGER DETECTION:**
   Identify correlations between internal physiology and external environment. Examples:
   - Elevated RR + High AQI = Environmental respiratory stress
   - Normal vitals + High cough count + Poor AQI = Early trigger exposure
   - Abnormal breathing pattern + Vocal inflammation = Subclinical progression

2. **SEVERITY CLASSIFICATION:**
   - CRITICAL (Score 70-100%): Immediate medical intervention needed
   - MODERATE (Score 40-69%): Early warning signs, proactive intervention required
   - LOW (Score 20-39%): Minor irregularities, continue monitoring
   - NORMAL (Score 0-19%): Healthy baseline, no concerns

3. **RISK SCORING LOGIC:**
   Calculate Total Risk Index (0-100%) using weighted factors:
   - Respiratory Rate abnormality: 30%
   - AI Pattern Classification: 25%
   - Cough frequency: 20%
   - Environmental triggers (AQI, PM2.5, Humidity, Pollen): 15%
   - Vocal biomarkers (if available): 10%

4. **PROACTIVE RECOMMENDATIONS:**
   Provide specific, actionable interventions:
   - Immediate actions (if critical)
   - Environmental avoidance strategies
   - Monitoring frequency
   - When to seek medical care

5. **RESPISTANT VOICE AGENT ALERT:**
   If risk is MODERATE or CRITICAL, generate a concise proactive warning message (2-3 sentences) that a voice agent would speak to the patient.

**OUTPUT FORMAT (Use Markdown):**

### 🔍 Clinical Findings
[2-3 sentences summarizing what the data reveals]

### ⚠️ Invisible Trigger Intersections
[Identify specific physiology-environment correlations]

### 📊 Risk Assessment
**Total Risk Index:** [X]%  
**Severity Level:** [CRITICAL/MODERATE/LOW/NORMAL]

**Risk Breakdown:**
- Respiratory abnormality contribution: [X]%
- AI pattern risk: [X]%
- Acoustic indicators: [X]%
- Environmental factors: [X]%

### 💊 Clinical Recommendations
1. [Immediate action if needed]
2. [Environmental avoidance strategy]
3. [Monitoring guideline]
4. [Medical escalation criteria]

### 🗣️ RespiStant Proactive Alert
[If MODERATE or CRITICAL: Write a 2-3 sentence voice agent message]
[If LOW/NORMAL: Write "No immediate alert required."]

Be medically precise, evidence-based, and actionable. Use clinical terminology but remain patient-friendly.
"""
    
    try:
        client = genai.Client(api_key=st.session_state.get('gemini_key', ''))
        response = client.models.generate_content(
            model='models/gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"⚠️ **Gemini Analysis Failed**\n\nError: {str(e)}\n\nPlease check:\n- API key validity\n- Internet connection\n- API quota limits"

# Collect all patient data for Gemini
voice_jitter = st.session_state.voice_data['jitter'] if st.session_state.voice_data else "Not tested"
voice_shimmer = st.session_state.voice_data['shimmer'] if st.session_state.voice_data else "Not tested"
voice_inflammation = "YES" if (st.session_state.voice_data and st.session_state.voice_data['inflammation_detected']) else "Not tested"

patient_data = {
    'respiratory_rate': st.session_state.rr,
    'heart_rate': st.session_state.hr,
    'ai_risk_score': st.session_state.risk_score,
    'cough_count': st.session_state.cough_count,
    'voice_jitter': f"{voice_jitter:.2f}%" if isinstance(voice_jitter, (int, float)) else voice_jitter,
    'voice_shimmer': f"{voice_shimmer:.2f}%" if isinstance(voice_shimmer, (int, float)) else voice_shimmer,
    'voice_inflammation': voice_inflammation,
    'aqi': st.session_state.current_aqi,
    'aqi_status': st.session_state.current_status,
    'pm25': f"{st.session_state.current_pm25:.1f}" if st.session_state.current_pm25 > 0 else "N/A",
    'humidity': st.session_state.humidity,
    'pollen': f"{st.session_state.pollen_index:.1f}" if st.session_state.pollen_index > 0 else "N/A"
}

# Check if meaningful data exists
has_data = (patient_data['respiratory_rate'] > 0 or 
            patient_data['cough_count'] > 0 or 
            patient_data['aqi'] > 1 or
            st.session_state.voice_data is not None)

if has_data:
    if st.button("🧠 Generate Gemini Clinical Report", type="primary", use_container_width=True):
        if not st.session_state.get('gemini_key'):
            st.error("⚠️ Please enter your Gemini API Key in the sidebar")
        else:
            with st.spinner("🔬 Gemini performing agentic data fusion..."):
                clinical_report = generate_gemini_clinical_report(patient_data)
                st.session_state.clinical_report = clinical_report
    
    # Display report if generated
    if 'clinical_report' in st.session_state:
        st.markdown(st.session_state.clinical_report)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Report (Markdown)",
                data=st.session_state.clinical_report,
                file_name=f"RespiSense_Clinical_Report.md",
                mime="text/markdown",
                use_container_width=True
            )
        with col2:
            if st.button("🔄 Clear Report", use_container_width=True):
                del st.session_state.clinical_report
                st.rerun()
else:
    st.info("📊 **MONITORING STATUS:** Upload sensor data, audio, voice recording, or check environmental data to enable clinical assessment.")

st.divider()
st.caption("⚠️ **Disclaimer:** This is an AI-assisted prototype for GDG TechSprint 2026 and is NOT intended for clinical diagnosis. Always consult healthcare professionals for medical decisions.")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json

    user_message = data.get("message")
    risk_level = data.get("risk_level")

    reply = chatbot_response(user_message, risk_level)

    return jsonify({"reply": reply})