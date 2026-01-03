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
Upload Phyphox CSV: Record chest accelerometer data (lying supine, 60 seconds)

Upload Audio: Record cough or breathing sounds

Upload Voice: Record sustained "Ahhh" sound (3-5 seconds)

Load Environmental Data: Fetch real-time air quality

Generate Report: Click to get Gemini clinical assessment

🏗️ Architecture

Input Layer → [Vitals | Cough | Voice | Environment]
           ↓
Gemini Fusion Layer → Clinical Reasoning
           ↓
Output → Risk Score + RespiStant Alerts

🛠️ Technologies
Frontend: Streamlit

ML Models: TensorFlow/Keras (MobileNetV2), Scikit-learn

LLM: Google Gemini 2.5 Flash

Signal Processing: SciPy, Librosa

APIs: OpenWeatherMap, Open-Meteo

📝 License
This project is a hackathon prototype for educational purposes.

👥 Team
XNN0V473R5! - GDG TechSprint 2026

⚠️ Disclaimer: This is an AI-assisted prototype and NOT for clinical diagnosis.

# Run application
streamlit run app.py
