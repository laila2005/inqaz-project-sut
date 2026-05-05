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
    3. Normalizes pixel values to the range `[-1.0, 1.0]` using `Rescaling(1./127.5, offset=-1.0)`. This is specifically required by MobileNetV2 because it was pre-trained on ImageNet with this exact range. Using the wrong range (e.g. `[0, 1]`) causes the model to see distorted colors and drastically reduces accuracy.
    4. Applies Data Augmentation (Random flips, rotation, zoom) to prevent the model from memorizing the images.

### The Brains (AI Models)
- **`src/models/cnn_scratch.py`**:
  - **What it does:** Contains the blueprint for our Custom Convolutional Neural Network built entirely from scratch.
  - **Why it matters:** Shows we understand how to build neural networks layer-by-layer (Conv2D -> BatchNorm -> ReLU -> MaxPooling).

- **`src/models/transfer_learning.py`**:
  - **What it does:** Loads a highly advanced, pre-trained Google model (`MobileNetV2`) but removes its top "classification" head. We then explicitly code our own custom head (`GlobalAveragePooling2D -> Dense(256) -> BatchNorm -> Dropout -> Dense(128) -> Dropout -> Output`).
  - **Why `GlobalAveragePooling2D` instead of `Flatten`?** `Flatten` converts the entire feature map into a massive 1D vector (millions of parameters), which causes severe overfitting on small datasets. `GlobalAveragePooling2D` instead computes the average of each feature map channel, collapsing it down to just 1,280 values. This drastically reduces the number of trainable parameters and forces the model to learn general patterns ("crumpled metal", "shattered glass") instead of memorizing exact pixel positions.
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
5. **GlobalAveragePooling2D & Dense Layers:** After the image has been broken down into a map of "features" (edges, tires, glass), `GlobalAveragePooling2D` computes the spatial average of each feature channel, compressing the information into a compact vector of 1,280 values. This is far more efficient than `Flatten` (which can produce millions of values and leads to overfitting). The `Dense` layers then act as the final decision-makers, looking at that compact vector and deciding if the features add up to a "Crash" or "Normal."
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
2. **Normalization:** We scaled pixel values to the range `[-1.0, 1.0]` using the formula `(pixel / 127.5) - 1.0`. This specific range is required by MobileNetV2 because that is how its original ImageNet weights were trained. Using a different range (like `[0, 1]`) would cause the pre-trained filters to see distorted input and fail to extract features correctly.
3. **Data Augmentation:** We applied random horizontal flips, rotations (20%), and zooming (20%) to the training data. This prevents overfitting by ensuring the model doesn't just memorize the exact training images.

### Q5: What loss function and final activation function did you use, and why?
**Answer:** 
- We used **Sigmoid** for the final activation function because this is a *Binary Classification* problem (Crash = 0, Normal = 1). Sigmoid perfectly outputs a probability between `0.0` and `1.0`.
- We used **Binary Crossentropy** as our loss function because it is the mathematical standard for penalizing wrong predictions in binary tasks. It measures the distance between the predicted probability and the actual true label.

### Q6: Explain your prediction logic in the application. How does the UI know it's a crash?
**Answer:** During our data audit, we discovered an anomaly in the Kaggle dataset: the folder labels were actually swapped during extraction (the `crash` folder contained normal images, and the `normal` folder contained crash images). Because our training pipeline automatically assigns classes alphabetically, `Class 0` was assigned to the `crash` folder (Normal images) and `Class 1` to the `normal` folder (Crash images). In `app.py`, the model outputs the probability of Class 1. Therefore, if the output probability is `> 0.5`, the system classifies it as a Crash and immediately triggers the emergency API. If it's `< 0.5`, it is flagged as Normal. Handling this anomaly demonstrated the importance of visual data validation.

### Q7: What is the F1-Score and why did you look at it instead of just Accuracy?
**Answer:** Accuracy can be misleading if a dataset is imbalanced. For example, if 90% of the images are "Normal," a model could just guess "Normal" every time and get 90% accuracy, while completely failing to detect crashes. The **F1-Score** is the harmonic mean of Precision and Recall. It proves that our model is actually good at correctly identifying crashes (Recall) without making too many false alarms (Precision).

### Q8: How did you deal with Overfitting?
**Answer:** We used four specific techniques to combat overfitting:
1. **Data Augmentation** during preprocessing.
2. **GlobalAveragePooling2D** instead of `Flatten`, which reduced the trainable parameters from ~16 million to ~1,280, preventing the model from memorizing pixel-level noise.
3. **Dropout layers** (0.5 and 0.3) in our Dense network, which randomly turns off neurons during training so the network doesn't rely too heavily on any single feature.
4. **Early Stopping** during training, which monitors the validation loss and automatically stops the training if the model stops improving, restoring the best weights.

### Q9: You discovered a dataset labeling anomaly. Explain what happened and how you handled it.
**Answer:** During a visual data audit, we discovered that the Kaggle dataset's folder labels were inverted — the `crash/` folder actually contained normal traffic images, and the `normal/` folder contained crash images. Since TensorFlow's `image_dataset_from_directory` assigns class indices alphabetically (`crash` = Class 0, `normal` = Class 1), the model learned that Class 0 = Normal and Class 1 = Crash. Instead of re-downloading and reorganizing 3,000 images, we adapted our application logic: in `app.py`, if the model output is `> 0.5` (Class 1), we display "Crash Detected." This is a real-world data engineering scenario — raw datasets are often messy, and a good engineer validates data visually before trusting folder names.

### Q10: Why did you replace `Flatten()` with `GlobalAveragePooling2D()` in the Transfer Learning model?
**Answer:** `Flatten()` converts every single value in the feature map grid into one enormous 1D vector. For a `7x7x1280` feature map from MobileNetV2, that produces `7 * 7 * 1280 = 62,720` values per image. When connected to a `Dense(256)` layer, that creates over **16 million trainable parameters**. With only ~2,100 training images, the model had far more parameters than data points, so it memorized the training set (overfitting) and failed on new images. `GlobalAveragePooling2D()` instead computes the average across each `7x7` spatial grid, producing only `1,280` values. This reduced the parameter count by over 99% and forced the model to learn meaningful, generalizable patterns like "crumpled metal" rather than memorizing exact pixel coordinates.

### Q11: Why must the normalization range match exactly between training and inference?
**Answer:** A pre-trained model like MobileNetV2 has internal filter weights that were calibrated during its original ImageNet training with inputs in the range `[-1.0, 1.0]`. If you feed it images in `[0, 1]` at training time, the filters see a completely different numerical distribution — it's like speaking a different language to the model. The model's feature extraction breaks down, and it learns weak, unreliable patterns. Furthermore, the normalization used during training **must be identical** to the normalization used during inference (in `app.py`). If training uses `[-1, 1]` but inference uses `[0, 1]`, the model receives inputs it has never seen before, and its predictions become random. We ensure consistency by using the exact formula `(pixel / 127.5) - 1.0` in both `preprocess.py` and `app.py`.

### Q12: How do you interpret the model's confidence score in the dashboard?
**Answer:** The model outputs a single floating-point number between `0.0` and `1.0` via the Sigmoid activation. Due to the dataset folder ordering, values closer to `1.0` indicate higher confidence that the image is a **Crash** (Class 1), and values closer to `0.0` indicate higher confidence that it is **Normal** (Class 0). We display the confidence as a percentage. For example, if the model outputs `0.87`, we display "Crash Detected — Confidence: 87.00%." If it outputs `0.15`, we display "Normal Situation — Confidence: 85.00%" (calculated as `1 - 0.15 = 0.85`). The 0.5 threshold is the standard decision boundary for binary classification with Sigmoid.
