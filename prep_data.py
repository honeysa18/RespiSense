import pandas as pd
import numpy as np
from scipy.signal import spectrogram
import cv2  # pip install opencv-python
import os

# --- CONFIGURATION ---
WINDOW_SEC = 10  # We cut signal into 10s chunks
files = ['HData.csv', 'KData.csv', 'RData.csv','abnormal_bradypnea.csv','abnormal_tachypnea.csv','abnormal_irregular.csv'] # Your 3 filenames
labels = [0, 0, 0, 1, 1, 1]  # 0=Normal, 1=Abnormal (You must decide this manually!)

X_train = []
y_train = []

def signal_to_image(signal, fs):
    """ Converts 1D signal to 224x224 Image for MobileNet """
    # 1. Spectrogram
    f, t, Sxx = spectrogram(signal, fs, nperseg=256, noverlap=128)
    # 2. Normalize to dB and 0-255 range
    S_db = 10 * np.log10(Sxx + 1e-10)
    S_min, S_max = S_db.min(), S_db.max()
    img = 255 * (S_db - S_min) / (S_max - S_min)
    img = img.astype(np.uint8)
    # 3. Resize to 224x224 (Model Input Size)
    img_resized = cv2.resize(img, (224, 224))
    # 4. Stack to make 3 channels (RGB)
    img_rgb = np.stack((img_resized,)*3, axis=-1)
    return img_rgb

# --- PROCESSING LOOP ---
for i, file_path in enumerate(files):
    try:
        # Load Data (Robust loading from your previous code)
        df = pd.read_csv(file_path)
        if 'Linear Acceleration z (m/s^2)' in df.columns:
            signal = df['Linear Acceleration z (m/s^2)'].values
        elif 'z' in df.columns:
            signal = df['z'].values
        else:
            signal = df.iloc[:, 3].values # Fallback
            
        t = df['Time (s)'].values
        fs = 1 / np.median(np.diff(t))
        
        # SLICING LOGIC (The Data Augmentation)
        samples_per_chunk = int(WINDOW_SEC * fs)
        
        # Loop through file and cut chunks
        # e.g., 0-10s, 10-20s, 20-30s...
        for start in range(0, len(signal) - samples_per_chunk, samples_per_chunk):
            chunk = signal[start : start + samples_per_chunk]
            
            # Convert chunk to Image
            img = signal_to_image(chunk, fs)
            
            # Add to Training Set
            X_train.append(img)
            y_train.append(labels[i]) # Inherit label from parent file
            
    except Exception as e:
        print(f"Skipping {file_path}: {e}")

# Convert to Numpy Arrays for AI
X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"✅ Data Preparation Complete!")
print(f"Original Files: {len(files)}")
print(f"Generated Samples: {X_train.shape[0]}") # Should be ~12
print(f"Training Shape: {X_train.shape}")


# --- IMPORT TENSORFLOW ---
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def train_model(X_data, y_data):
    print("\n🚀 Starting Transfer Learning...")

    # 1. Load the Pre-Trained Brain (MobileNetV2)
    # include_top=False means we chop off the head (ImageNet classifier)
    # input_shape=(224, 224, 3) matches the images we made
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # 2. Freeze the Base
    # We don't want to retrain MobileNet, just our new layers
    base_model.trainable = False 

    # 3. Add Your "Medical" Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Flatten 
    x = Dense(16, activation='relu')(x) # Intermediate layer
    predictions = Dense(1, activation='sigmoid')(x) # Output: 0 to 1

    # 4. Compile
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    # 5. Train!
    # validation_split=0.2 means it uses 20% of data to test itself while training
    history = model.fit(X_data, y_data, epochs=10, batch_size=4, validation_split=0.2)
    
    # 6. Save the Model
    model.save('respi_model.h5')
    print("\n✅ SUCCESS: Model saved as 'respi_model.h5'")
    return history

# --- EXECUTE TRAINING ---
if len(X_train) > 0:
    train_model(X_train, y_train)
else:
    print("❌ Error: No data was generated. Check your CSV filenames.")