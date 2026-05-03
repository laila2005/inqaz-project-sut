# Inqaz AI Rescue System - Complete Documentation

Welcome to the documentation for the **Inqaz AI Rescue System**. This document is designed to be easy to understand, breaking down what the app does, how the AI concepts work, and how the underlying code processes data.

---

## 1. Project Summary & Achievements

### Dataset Used
- **Name:** Car Crash or Collision Prediction Dataset
- **Author:** Md. Fahim Bin Amin
- **URL:** [https://www.kaggle.com/datasets/mdfahimbinamin/car-crash-or-collision-prediction](https://www.kaggle.com/datasets/mdfahimbinamin/car-crash-or-collision-prediction)

### What We Have Done
1. **Data Engineering:** Extracted and cleaned a raw Kaggle dataset, programmatically mapping image IDs to their correct labels via an Excel database.
2. **Data Balancing:** Created a perfectly balanced subset of exactly 3,000 images (1,500 crash, 1,500 normal) to prevent AI bias.
3. **Model Engineering:** Designed and built two separate neural networks: a Custom CNN built layer-by-layer, and a MobileNetV2 transfer learning model.
4. **Full-Stack Dashboard:** Developed a real-time web application using Streamlit that automatically triggers mock emergency API responses (Police, Ambulance, GPS) when a crash is detected.

### Current Accuracy Reached
- **Transfer Learning Model (MobileNetV2):** ~70% overall accuracy (F1-score of 0.70) on entirely unseen test data.
- **Custom CNN:** ~65% overall accuracy.
*(Note: These are strong baselines given the relatively small training subset of 3,000 images. The models correctly identify the majority of severe crash scenarios).*

---

## 2. What the App Does (Overview)
The **Inqaz AI Rescue System** is an AI-powered emergency response application designed for Egypt. Its main goal is to automatically detect severe car crashes from images and instantly alert the necessary authorities to save lives.

### The User Flow:
1. **Image Upload:** A user (or a camera system) uploads a photo of a road or intersection to the dashboard.
2. **AI Analysis:** The system analyzes the image in real-time to determine if a car crash has occurred.
3. **Automated Response:** 
   - If **Normal**, the system does nothing and confirms the situation is safe.
   - If a **Crash is Detected**, the system automatically triggers a sequence of emergency actions:
     - Extracts precise GPS coordinates.
     - Sends an alert to the Ministry of Interior (Police - 122).
     - Dispatches the nearest Ambulance (123).
     - Attaches the photo evidence to the dispatch center.

---

## 3. The Concept: How the AI Works
At the heart of the system is **Computer Vision**, a branch of Artificial Intelligence that allows computers to "see" and understand images. 

Specifically, this project uses a technique called **Binary Image Classification**. This means the AI is trained to sort images into exactly two categories:
- `Class 0`: **Crash** (An accident has occurred).
- `Class 1`: **Normal** (Standard traffic or empty road).

### The Two Brains (Models)
We built two different AI "brains" (Neural Networks) to solve this problem:
1. **Custom CNN (Built from Scratch):** A Convolutional Neural Network built block-by-block. It learns basic shapes, edges, and textures from the ground up to identify crashed cars.
2. **Transfer Learning (MobileNetV2):** A highly advanced model created by Google that has already seen millions of everyday images. We took this "smart" model and retrained its final layers specifically to recognize car crashes. This model generally performs better because it already knows how to process complex visuals.

---

## 4. How the Data Processing Works
Before the AI can look at a photo, the photo must be translated into numbers. Here is the exact pipeline of how processing works from start to finish:

### A. Training Preprocessing (Teaching the AI)
When we are teaching the AI, we use the `src/preprocess.py` script:
- **Resizing:** Every camera takes photos at different sizes. The code squashes every image into a perfect square: `224x224 pixels`.
- **Normalization:** Computer screens use colors ranging from 0 to 255. The code divides these numbers by 255 so the AI only has to do math with small numbers between `0.0` and `1.0`.
- **Data Augmentation:** To make the AI smarter, the code slightly rotates, flips, and zooms the training images. This forces the AI to recognize a crash even if the camera is upside down or far away.

### B. Live App Processing (Inference)
When you upload an image to the live Streamlit dashboard (`app.py`), the following happens:
1. The image is loaded into memory using the `PIL` library.
2. The image is resized to `224x224` to match exactly what the AI expects.
3. The image colors are converted into an array of numbers and divided by 255 (`img_array / 255.0`).
4. An extra "dimension" is added because the AI expects a *batch* of images, even if we are only giving it one.
5. The processed numbers are fed into the AI model (`model.predict()`).
6. The AI outputs a single decimal number between `0.0` and `1.0`.
   - If the number is **less than 0.5**, it is closer to Class 0 (Crash).
   - If the number is **greater than 0.5**, it is closer to Class 1 (Normal).

---

## 5. Code Structure Explained
If you are looking at the project files, here is what each script does:

* `app.py`: The main user interface. It uses the `Streamlit` library to create the web page, handle image uploads, process the image, and display the alerts.
* `src/train.py`: The script that was run once to teach the AI. It loops through all 3,000 dataset images and saves the final "smart" models into the `results/saved_models/` folder.
* `src/evaluate.py`: The testing script. It gives the AI a pop quiz on images it has never seen before, and generates the charts (Confusion Matrix, ROC Curve) to prove to your university how accurate it is.
* `src/preprocess.py`: The factory line. It prepares, cleans, and augments the images before they go into the neural networks.
* `src/models/cnn_scratch.py`: Contains the raw blueprint for the custom-built neural network.
* `src/models/transfer_learning.py`: Contains the blueprint for importing Google's MobileNetV2 and attaching our custom classification head to it.

---

### Summary
The system successfully bridges complex deep learning concepts with a practical, user-friendly interface. By utilizing strict image preprocessing and powerful convolutional neural networks, the app is able to ingest visual data and make life-saving automated decisions within seconds.
