# Inqaz AI - Final Defense & Viva Study Guide

This document is the ultimate study guide for the entire team to prepare for the final project discussion (viva). It breaks down exactly what every file does, explains how the CNN works in simple terms, and provides complete answers to the questions the professor is likely to ask.

---

## 1. File-by-File Codebase Breakdown

Every team member must understand what each script does. Here is the exact pipeline:

### Data Preparation
- **`src/setup_dataset.py`**: 
  - **What it does:** Downloads the raw Car Crash dataset directly from Kaggle and organizes the images into two folders: `data/raw/crash` and `data/raw/normal`. 
  - **Why it matters:** Proves we didn't use a built-in "toy" dataset like MNIST. We handled raw files manually.

- **`src/preprocess.py`**:
  - **What it does:** Prepares the images before they go into the AI.
  - **Key steps:** 
    1. Resizes all images to exactly `224x224` pixels.
    2. Splits the data (70% Train, 15% Validation, 15% Test).
    3. Normalizes pixel values (divides by 255 to make them between 0 and 1).
    4. Applies Data Augmentation (Random flips, rotation, zoom) to prevent the model from memorizing the images.

### The Brains (AI Models)
- **`src/models/cnn_scratch.py`**:
  - **What it does:** Contains the blueprint for our Custom Convolutional Neural Network built entirely from scratch.
  - **Why it matters:** Shows we understand how to build neural networks layer-by-layer (Conv2D -> BatchNorm -> ReLU -> MaxPooling).

- **`src/models/transfer_learning.py`**:
  - **What it does:** Loads a highly advanced, pre-trained Google model (`MobileNetV2`) but removes its top "classification" head. We then explicitly code our own custom head (`Flatten -> Dense -> Dropout -> Output`).
  - **Why it matters:** Fulfills the requirement of doing "manual transfer learning" without drag-and-drop. It shows we know how to freeze and unfreeze specific layers of a massive model.

### Training & Testing
- **`src/train.py`**:
  - **What it does:** The script that actually trains both models. It feeds the training images to the networks, calculates the loss, updates the weights, and saves the "smart" models as `.h5` files in the `results/` folder. It also draws the Training vs. Validation graphs to prove the models didn't overfit.

- **`src/evaluate.py`**:
  - **What it does:** Gives the trained AI a "pop quiz" on the 15% Test Data (images it has never seen before). It generates the Classification Report (Accuracy, Precision, Recall, F1), the Confusion Matrix Heatmap, and the ROC Curve.

### The Deployment
- **`app.py`**:
  - **What it does:** The actual Streamlit Web Application (The "Inqaz" Dashboard). It accepts an image upload, processes it identically to the training data, feeds it to the best model, and then simulates the emergency response (GPS coordinates, Ambulance, Police) if a crash is detected.

---

## 2. How the CNN Works (Simply Explained)

If the professor asks *"How does your CNN actually work?"*, explain it like an assembly line in a factory:

1. **Convolution (`Conv2D`):** The network slides small "filters" over the image. In the first layers, these filters look for basic things like straight lines, edges, and colors. In deeper layers, they combine those edges to find complex shapes like "tires," "shattered glass," or "dented metal."
2. **Activation (`ReLU`):** This is a math function that deletes any negative numbers in the image data, replacing them with zero. This helps the network focus only on the important features and speeds up calculation.
3. **Pooling (`MaxPooling2D`):** This step shrinks the image size by keeping only the most important, sharpest features in each small area. It helps the network recognize a crashed car whether it's in the top-left corner or the bottom-right corner.
4. **Batch Normalization:** This stabilizes the learning process. It ensures the data passing between layers doesn't get too large or too small, making the training much faster and more stable.
5. **Flatten & Dense Layers:** After the image has been broken down into a map of "features" (edges, tires, glass), `Flatten` turns it into a single long list of numbers. The `Dense` layers act as the final decision-makers, looking at that list and deciding if the features add up to a "Crash" or "Normal."
6. **Output Layer (`Sigmoid`):** Because this is a Binary Classification (two choices), the final output is a single neuron with a Sigmoid function. It squashes the final math into a probability between `0.0` and `1.0`.

---

## 3. Professor Q&A Prep

### Q1: Why did you choose to use both a Custom CNN and Transfer Learning (MobileNetV2)?
**Answer:** We built a Custom CNN to demonstrate our foundational understanding of deep learning architectures, building layers from scratch. We then used MobileNetV2 to compare our custom model against an industry-standard. MobileNetV2 was specifically chosen because it is extremely lightweight and fast, making it ideal for real-time deployments like our emergency dashboard.

### Q2: What exactly is "Transfer Learning" and why is it beneficial here?
**Answer:** Transfer learning means taking a model that has already been trained on millions of images (like ImageNet) and reusing its "knowledge." Instead of our AI having to learn what an edge or a shadow is from scratch, it already knows. We just removed its original final layer and added our own custom head to teach it to specifically recognize "Crash" vs "Normal."

### Q3: Explain your two-phase training approach for Transfer Learning.
**Answer:** In the first phase, we "froze" the base MobileNetV2 model and only trained our new custom head so it could adjust to the initial data. In the second phase, we "unfroze" the top 20 layers of the base model and trained the whole thing again at a very low learning rate. This "fine-tuning" allows the pre-trained weights to adapt specifically to our car crash textures without destroying their original knowledge.

### Q4: How did you preprocess your dataset before training?
**Answer:** We applied three main steps:
1. **Resizing:** All images were resized to exactly `224x224` pixels because our neural network architectures require a fixed input size.
2. **Normalization:** We scaled the pixel values by dividing by `255`. This converts pixel values from `0-255` to a range of `0.0 to 1.0`, which helps the neural network converge much faster mathematically.
3. **Data Augmentation:** We applied random horizontal flips, rotations (20%), and zooming (20%) to the training data. This prevents overfitting by ensuring the model doesn't just memorize the exact training images.

### Q5: What loss function and final activation function did you use, and why?
**Answer:** 
- We used **Sigmoid** for the final activation function because this is a *Binary Classification* problem (Crash = 0, Normal = 1). Sigmoid perfectly outputs a probability between `0.0` and `1.0`.
- We used **Binary Crossentropy** as our loss function because it is the mathematical standard for penalizing wrong predictions in binary tasks. It measures the distance between the predicted probability and the actual true label.

### Q6: Explain your prediction logic in the application. How does the UI know it's a crash?
**Answer:** Because our raw dataset folders were named alphabetically (`crash` and `normal`), the training pipeline automatically assigned `Class 0` to Crash and `Class 1` to Normal. In `app.py`, when a user uploads an image, the model outputs a probability. If the output probability is `< 0.5`, the system classifies it as a Crash (Class 0) and immediately triggers the automated emergency API sequence. If it's `> 0.5`, it is flagged as a Normal situation.

### Q7: What is the F1-Score and why did you look at it instead of just Accuracy?
**Answer:** Accuracy can be misleading if a dataset is imbalanced. For example, if 90% of the images are "Normal," a model could just guess "Normal" every time and get 90% accuracy, while completely failing to detect crashes. The **F1-Score** is the harmonic mean of Precision and Recall. It proves that our model is actually good at correctly identifying crashes (Recall) without making too many false alarms (Precision).

### Q8: How did you deal with Overfitting?
**Answer:** We used three specific techniques to combat overfitting:
1. **Data Augmentation** during preprocessing.
2. **Dropout layers** (0.5 and 0.3) in our Dense network, which randomly turns off neurons during training so the network doesn't rely too heavily on any single feature.
3. **Early Stopping** during training, which monitors the validation loss and automatically stops the training if the model stops improving, restoring the best weights.
