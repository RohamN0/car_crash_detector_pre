# 🚗 Real-Time Car Crash Detection

This project performs automatic accident detection from videos using a
Streamlit interface and a deep-learning pipeline that extracts:

-   RGB frames
-   Optical-flow divergence masks
-   Temporal window around the event

The final model predicts whether the video contains a **CAR CRASH** or
**SAFE** sequence.

## 📁 Project Structure

    project/
    ├── app.py
    ├── model.py
    ├── modele.keras
    ├── data/
    │   └── train.csv
    └── videos/
        ├── video_matrices/
        └── masked_video_matrices/

## 🧠 Model Information

This project uses the GRU + ResNet50 model from:\
https://github.com/saraM0radi/Sentiment_Analysis\
Accuracy: **85%**\
You may replace this with any other Keras model.

## ⚙️ Requirements

    pip install streamlit tensorflow keras torch torchvision opencv-python numpy pandas

## ▶️ Running the App

    streamlit run app.py

## 🧩 Pipeline Overview

### 1) Frame Extraction

-   Extracts a 2-second window around crash time (for positive samples).
-   For negative samples, extracts a random 2-second segment.

### 2) Frame Preprocessing

-   Resize to **224×224** on GPU (PyTorch).
-   Pad/truncate to **10 frames**.

### 3) Optical-Flow Divergence Mask

For each frame pair, the system: - Computes optical-flow gradients. -
Generates: - Motion magnitude\
- Motion angle\
- Divergence map

### 4) Model Prediction

Outputs: - 🚨 **CRASH** - ✅ **SAFE**
