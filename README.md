# 🫁 RespiSense AI - Intelligent Respiratory Profiler

**GDG TechSprint 2026 Submission**

RespiSense AI is a multi-modal respiratory monitoring system that detects "invisible trigger intersections" between internal physiology and external environment using Gemini-powered data fusion.

## 🌟 Features

### Layer 1: Multi-Modal Sensing
- **Kinematic Vital Monitor**: Seismocardiography-based RR & HR extraction from smartphone accelerometer
- **Acoustic Biomarker Engine**: CNN-based cough detection and breathing pattern classification
- **Vocal Resonance Analyzer**: Jitter/Shimmer analysis + ML classification for airway inflammation
- **Environmental Radar**: Real-time AQI, PM2.5, humidity, pollen monitoring

### Layer 2: Gemini Clinical Fusion
- Agentic data fusion with Gemini 2.5 Flash
- Invisible trigger correlation (physiology × environment)
- Risk stratification with RespiStant proactive alerts

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/RespiSense-AI.git
cd RespiSense-AI

# Install dependencies
pip install -r requirements.txt

```

🔑 Setup
Get Gemini API Key from Google AI Studio

Enter API key in the sidebar when app launches

📊 Usage
Upload Phyphox CSV: Record chest accelerometer data (lying supine, 60 seconds) - Normal and Abnormal Breathe

Upload Audio: Record cough or breathing sounds - Cough and Non Cough files

Upload Voice: Record sustained "Ahhh" sound (3-5 seconds) - Audio Samples

Load Environmental Data: Fetch real-time air quality - Fetched from APIs

Generate Report: Click to get Gemini clinical assessment

🏗️ Architecture

Input Layer → [Vitals | Cough | Voice | Environment]
           ↓
Gemini Fusion Layer → Clinical Reasoning
           ↓
Output → Risk Score + RespiStant Alerts

🛠️ Technologies

Frontend & Deployment : Streamlit, Streamlit Community Cloud, HTML/CSS/JavaScript

Machine Learning & AI : TensorFlow 2.17.0 / Keras, MobileNetV2, Scikit-learn, Google Gemini 2.0 Flash

Signal Processing & Audio Analysis : SciPy, Librosa, Parselmouth (Praat), NumPy, Pandas

Sensor : Smartphone accelerometer (seismocardiography)

External APIs & Environmental Context : Google Maps API, Google Geolocation API, Google Weather API, Google Air Quality API, Google Pollen API

Data Visualization : Matplotlib, OpenCV, Streamlit Charts

Medical Data Sources & Training Datasets : 
1. Respiratory Sound Database - Kaggle/ICBHI 2017 Scientific Challenge (920 annotated audio samples)
2. COVID-19 Cough Audio Dataset - Open-source respiratory distress recordings
3. Voice Pathology Database - Saarbrucken Voice Database (healthy vs. pathological voice recordings)
4. Custom Phyphox Accelerometer Data - Self-collected seismocardiography recordings for model validation
5. Clinical Guidelines - WHO respiratory rate norms, ATS/ERS voice quality standards

Development Tools : Python 3.11, Git/GitHub, Google Colab, Joblib, VS Code

📝 License
This project is a hackathon prototype for educational purposes.

👥 Team
XNN0V473R5! - GDG TechSprint 2026

⚠️ Disclaimer: This is an AI-assisted prototype and NOT for clinical diagnosis.
# Run application
streamlit run app.py
