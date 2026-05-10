# Inqaz AI Rescue System - Complete Documentation

Welcome to the documentation for the **Inqaz AI Rescue System**. This document is designed to be easy to understand, breaking down what the app does, how the AI concepts work, and how the underlying code processes data.

---

## 1. Project Summary & Achievements

### Dataset Used
- **Name:** Car Crash Dataset
- **Author:** Vishnu606
- **URL:** [https://www.kaggle.com/datasets/vishnu606/car-crash-dataset](https://www.kaggle.com/datasets/vishnu606/car-crash-dataset)
- **Total Images Used:** 3,000 (1,500 Crash + 1,500 Normal) — balanced to prevent AI bias.

### What We Have Done
1. **Data Engineering:** Downloaded a raw Kaggle dataset using the `kagglehub` library, programmatically sorting images into `crash/` and `normal/` folders based on their directory structure.
2. **Data Balancing:** Created a perfectly balanced subset of exactly 3,000 images (1,500 crash, 1,500 normal) to prevent AI bias.
3. **Model Engineering:** Designed and built two separate neural networks:
   - A **Custom CNN** built layer-by-layer from scratch.
   - A **Transfer Learning model** using the MobileNetV2 backbone with a custom classification head.
4. **Explainable AI (Grad-CAM):** Implemented visual heatmaps to show exactly what parts of an image the AI is looking at to make its decision.
5. **Full-Stack Deployment:** Developed a premium dark-mode web application using Streamlit. Integrated a Live Camera feed and deployed the entire system to **Streamlit Community Cloud** for global accessibility.

### Current Accuracy Reached
- **Transfer Learning Model (MobileNetV2):** ~68% overall accuracy (F1-score of 0.68) on entirely unseen test data.
- **Custom CNN:** ~63% overall accuracy.

---

## 2. What the App Does (Overview)
The **Inqaz AI Rescue System** is an AI-powered emergency response application. Its main goal is to automatically detect severe car crashes from images or live camera feeds and instantly alert the necessary authorities.

### The User Flow:
1. **Input:** A user uploads a photo OR uses the Live Camera integration.
2. **AI Analysis:** The system analyzes the image in real-time. If a crash is detected, it generates a **Grad-CAM Heatmap** overlaid on the image to visually highlight the vehicle damage.
3. **Automated Response:** 
   - If **Normal**, the system does nothing and confirms the situation is safe.
   - If a **Crash is Detected**, the system extracts precise GPS coordinates, alerts the Ministry of Interior (Police - 122), dispatches an Ambulance (123), and attaches the heatmap evidence.

---

## 3. AI Concepts Glossary (Explained Simply)

This section explains the core concepts used in this project:

### Computer Vision & Binary Classification
- **Computer Vision:** A field of AI that allows computers to "see" and interpret digital images.
- **Binary Classification:** A task where the AI must choose between exactly two options. In our project, it sorts images into `Class 0` (Crash) or `Class 1` (Normal).

### Convolutional Neural Network (CNN)
A type of AI designed specifically for images. It works by sliding small "filters" over an image to detect patterns. 
- Early layers detect simple things like straight lines, edges, and shadows.
- Deeper layers combine those lines into complex shapes like "tires," "shattered glass," or "dented metal."

### Transfer Learning
Instead of teaching an AI from scratch (like teaching a baby), Transfer Learning takes an "adult" AI that has already studied millions of images (like Google's MobileNetV2) and slightly retrains its final brain cells to recognize a specific new task (like car crashes). This saves time and drastically improves accuracy.

### Overfitting & Global Average Pooling
- **Overfitting:** This happens when an AI just *memorizes* the training images instead of actually learning how to spot a crash. It scores 100% on the training test but fails on real-world photos.
- **Global Average Pooling:** A technique we used to prevent overfitting. Instead of taking every single pixel of feature data (which creates millions of parameters and leads to memorization), it calculates the "average" of the features, summarizing the image into a tiny, generalized list of numbers.

### Data Augmentation & Normalization
- **Normalization:** Computers don't see colors; they see numbers from 0 to 255. Normalization shrinks these numbers down to a small range like `[-1.0, 1.0]`. This makes the math easier and faster for the AI.
- **Data Augmentation:** To make the AI smarter, we slightly rotated, flipped, and zoomed the training images. This forces the AI to recognize a crash even if the camera is upside down, effectively creating thousands of "new" training examples from the original data.

### Explainable AI (Grad-CAM)
AI is often considered a "black box" because humans don't know *why* it makes a decision. **Grad-CAM (Gradient-weighted Class Activation Mapping)** fixes this. By tracing the mathematical gradients backward from the final decision, Grad-CAM draws a thermal heatmap over the original image showing exactly which pixels (e.g., a crushed bumper) most strongly convinced the AI that a crash occurred.

---

## 4. How the Data Processing Works

### A. Training Preprocessing (Teaching the AI)
When we are teaching the AI, we use the `src/preprocess.py` script:
1. **Resizing:** Squashes every image into a perfect square: `224x224 pixels`.
2. **Normalization:** Uses `Rescaling(1./127.5, offset=-1.0)` to force pixel values into the `[-1.0, 1.0]` range required by MobileNetV2.
3. **Data Augmentation:** Randomly flips images horizontally, rotates up to 20%, and zooms up to 20%.
4. **Data Split:** 70% Training (~2,100 images), 15% Validation (~450 images), 15% Test (~450 images).

### B. Live App Processing (Inference)
When you upload an image to the live Streamlit dashboard (`app.py`):
1. The image is resized to `224x224`.
2. The image colors are converted into the `[-1.0, 1.0]` range using `(pixel / 127.5) - 1.0`.
3. An extra "batch dimension" is added.
4. The AI model predicts a single decimal number between `0.0` and `1.0`.
   - **Less than 0.5** = Class 0 (Crash).
   - **Greater than or equal to 0.5** = Class 1 (Normal).
5. If it's a crash, the Grad-CAM algorithm calculates the heatmap and overlays it in the dashboard.

---

## 5. Code Structure Explained

* `app.py`: The Streamlit web application. Handles the UI, Live Camera, model inference, and Grad-CAM visualization.
* `src/setup_dataset.py`: Downloads the Kaggle car crash dataset using `kagglehub` and sorts images into `data/raw/crash/` and `data/raw/normal/`.
* `src/preprocess.py`: Prepares, cleans, and augments the images before they go into the neural networks.
* `src/models/cnn_scratch.py`: The blueprint for the custom-built CNN (4 Conv blocks → Flatten → Dense(512) → Dense(128) → Dense(1)).
* `src/models/transfer_learning.py`: The blueprint for importing MobileNetV2 and attaching our custom classification head (GlobalAveragePooling2D → Dense(256) → BatchNorm → ReLU → Dropout(0.5) → Dense(128) → BatchNorm → ReLU → Dropout(0.3) → Dense(1)).
* `src/train.py`: Trains both AI models with Early Stopping and saves the best weights to `results/saved_models/`.
* `src/evaluate.py`: Tests the AI on unseen images and generates the Classification Report, Confusion Matrix, and ROC Curves.
