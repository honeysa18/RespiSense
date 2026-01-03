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

# Run application
streamlit run app.py
