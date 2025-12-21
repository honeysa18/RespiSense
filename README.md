# 🫁 RespiSense AI
**A Multi-Modal Respiratory Health Intelligence System**

RespiSense is an integrated diagnostic dashboard designed to provide real-time respiratory risk assessment. By correlating physical chest motion, acoustic symptoms, and environmental triggers, it provides a holistic view of a patient's pulmonary health.



## 🌟 Key Features
- **Kinematic Monitoring:** Uses smartphone accelerometers to extract Respiratory Rate (RR) and Heart Rate (SCG) through Z-axis vibration analysis.
- **Acoustic Fingerprinting:** A Random Forest classifier trained on MFCC features to detect and count cough events in real-time audio.
- **AI-Driven Diagnosis:** A Deep Learning model (MobileNetV2 architecture) that analyzes respiratory spectrograms for abnormal breathing patterns.
- **Environmental Context:** Live Air Quality Index (AQI) integration via OpenWeatherMap API to correlate external pollutants (PM2.5) with symptom flare-ups.

## 🏗️ Project Architecture
The project is structured modularly for scalability:
- `/Modules`: Contains the core logic for signal processing (`RR_HR.py`), cough detection (`cough.py`), and AQI retrieval (`aqi.py`).
- `/Models`: Contains the pre-trained TensorFlow `.h5` model and the Scikit-Learn `.joblib` classifier.
- `app.py`: The Streamlit-based interactive dashboard.



## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- An OpenWeatherMap API Key (Optional: Use built-in mock data for demo)

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/honeysa18/RespiSense.git](https://github.com/honeysa18/RespiSense.git)
   cd RespiSense
2. Install Dependencies
   pip install -r requirements.txt
3. Run the application:
   python -m streamlit run app.py
   
## 📊 How it Works
Signal Processing: We apply a 4th-order Butterworth Bandpass filter to isolate the Respiratory signal (0.1–0.5 Hz) and the Cardiac signal (0.8–2.5 Hz).

Transfer Learning: 1D sensor data is converted into 2D Spectrograms to leverage Computer Vision for respiratory distress detection.

Data Fusion: Our "Master Risk Index" uses weighted correlation logic to provide a final clinical verdict.

## 🛠️ Tech Stack
Frontend: Streamlit

Deep Learning: TensorFlow / Keras

Machine Learning: Scikit-Learn

Audio Analysis: Librosa

Signal Processing: SciPy / NumPy
   ```bash
   git clone [https://github.com/honeysa18/RespiSense.git](https://github.com/honeysa18/RespiSense.git)
   cd RespiSense
