import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# --- 1. CONFIGURATION ---
BASE_DATA_DIR = "Data"
OUTPUT_DIR = "Models"
TARGET_SIZE = (224, 224)  # Required for MobileNetV2
CATEGORIES = {
    "Normal_breath": 0,
    "Abnormal_breath": 1
}

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 2. SIGNAL PROCESSING ---
def create_spectrogram(signal, fs=50):
    """Converts 1D sensor signal to 2D spectrogram image"""
    # Window: 2-3 seconds to capture 0.5-1 full breath
    window_length = 128  # 2.56 seconds at 50 Hz
    overlap = 96         # 75% overlap for smooth transitions
    f, t, Sxx = spectrogram(signal, fs, nperseg=window_length, noverlap=overlap)
    S_db = 10 * np.log10(Sxx + 1e-10) # Convert to Decibels
    
    # Normalize to 0-255 for Image Processing
    S_min, S_max = S_db.min(), S_db.max()
    if S_max > S_min:
        img = 255 * (S_db - S_min) / (S_max - S_min)
    else:
        img = np.zeros(S_db.shape)
        
    # Resize and convert to 3-channel (RGB) for AI model compatibility
    img_resized = cv2.resize(img.astype(np.uint8), TARGET_SIZE)
    img_rgb = np.stack((img_resized,)*3, axis=-1)
    return img_rgb

# --- 3. DATA PREPARATION ---
def prepare_respi_data():
    all_images = []
    all_labels = []

    print(f"🚀 Starting Data Prep in: {BASE_DATA_DIR}")

    for category, label in CATEGORIES.items():
        folder_path = os.path.join(BASE_DATA_DIR, category)
        
        if not os.path.exists(folder_path):
            print(f"⚠️ Warning: Folder not found: {folder_path}")
            continue

        print(f"📂 Processing Category: {category}")
        
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(folder_path, filename)
                try:
                    df = pd.read_csv(file_path)
                    # Support both standard name and short 'z'
                    sig_col = next((col for col in df.columns if 'z' in col.lower()), None)
                    
                    if sig_col:
                        signal = df[sig_col].values
                        # Ensure at least 10 seconds of data at 50Hz
                        if len(signal) >= 500:
                            mid = len(signal) // 2
                            chunk = signal[mid-250 : mid+250]
                            
                            spec_img = create_spectrogram(chunk)
                            all_images.append(spec_img)
                            all_labels.append(label)
                            print(f"  ✅ Processed {filename}")
                except Exception as e:
                    print(f"  ❌ Error processing {filename}: {e}")

    return np.array(all_images), np.array(all_labels)


# --- 5. MODEL TRAINING ---
def train_respi_model(X, y):
    print("\n🧠 Building and Training the AI Model...")
    
    # Use Transfer Learning with MobileNetV2
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False # Freeze pretrained layers

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid') # Binary Output: 0 or 1
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Augmentation to improve generalization
    datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.05,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest'
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Split for validation
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print("🏁 Starting training (10 Epochs)...")
    model.fit(
        datagen.flow(X_train, y_train, batch_size=4), # Use augmented data
        epochs=20,
        validation_data=(X_val, y_val),
        callbacks=[early_stop]
    )

    # --- SAVE THE FINAL MODEL ---
    model_path = os.path.join(OUTPUT_DIR, "respi_model.h5")
    model.save(model_path)
    print(f"\n✅ SUCCESS: Model saved as '{model_path}'")

#--- 4. VISUALIZATION ---
def show_preview(X, y):
    """Displays the first generated spectrogram to verify logic"""
    if len(X) >0:
        plt.figure(figsize=(8, 5))
        plt.imshow(X[3])
        label_text = "Abnormal" if y[0] == 1 else "Normal"
        plt.title(f"AI Input Preview (Label: {label_text})")
        plt.axis('off')
        print("🖼️ Showing spectrogram preview... (Close the window to continue to training)")
        plt.show()

# --- 6. MAIN EXECUTION ---
if __name__ == "__main__":
    X, y = prepare_respi_data()
    
    if len(X) > 0:
        print(f"📊 Dataset Ready: {len(X)} samples found.")
        
        # 1. Visualize one to make sure it looks correct
        show_preview(X, y)
        
        # 2. Train the model to generate the .h5 file
        train_respi_model(X, y)
    else:
        print("\n❌ FATAL: No data found. Ensure your 'Data' folder has CSVs.")