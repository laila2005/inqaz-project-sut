# Inqaz - Final Project Report Outline

Use this outline to write your PDF or Word report to get full marks on the Documentation (5 marks).

## 1. Project Overview & Task Definition
- **Idea:** Egypt's AI Emergency Rescue System (Inqaz).
- **Goal:** Detect car crashes from images to automatically alert authorities (MOI/Ambulance).
- **Task:** Binary Image Classification (Crash vs. Normal Traffic).

## 2. Dataset Description
- **Source:** Kaggle (Car Crash Dataset). Mention the URL.
- **Size:** State the exact number of images (ensure it's > 1000).
- **Structure:** Raw images stored locally without pre-computation.

## 3. Preprocessing Steps
*Explain what you did in `src/preprocess.py` and why:*
- **Resize:** Resized all images to a uniform 224x224 pixels to fit the model input layer.
- **Normalization:** Scaled pixel values by `1/255` to convert them from [0, 255] to [0, 1] for faster convergence.
- **Data Augmentation:** Applied random horizontal flips, rotation (20%), and zoom (20%) to artificially expand the dataset and prevent the model from overfitting to specific orientations.
- **Data Split:** 70% Training, 15% Validation, 15% Testing using a random seed to prevent data leakage.

## 4. Model Architectures
### Model 1: Scratch CNN
- Built layer by layer.
- 4 Convolutional Blocks containing: `Conv2D -> BatchNorm -> ReLU -> MaxPooling2D`.
- Explained use of `BatchNormalization` for stability and `Dropout` (0.5 and 0.3) to reduce overfitting.
- Output layer: 1 Neuron with `Sigmoid` activation.

### Model 2: Transfer Learning (MobileNetV2)
- Built manually without using one-line wrapper functions.
- Used MobileNetV2 pretrained on ImageNet (excluding the top head).
- Explicitly added custom head: `Flatten() -> Dense(256) -> Dropout(0.5) -> Dense(1, Sigmoid)`.
- Discussed the Two-Phase training strategy: Frozen base layers first, followed by fine-tuning the top 20 layers.

## 5. Evaluation Results
*Take screenshots of the PNGs generated in the `/results/` folder and paste them here.*
- **Classification Report:** Show Accuracy, Precision, Recall, and F1-Score for both models.
- **Confusion Matrix:** Insert the generated Heatmaps and explain True Positives, False Positives, etc.
- **ROC/AUC:** Insert the ROC curve and explain what the AUC value signifies.
- **Loss/Accuracy Curves:** Show the training vs validation graphs to prove the models didn't overfit.

## 6. Model Comparison Table
| Metric | Scratch CNN | Transfer Learning |
|--------|-------------|-------------------|
| Train Accuracy | % | % |
| Val Accuracy | % | % |
| Test Accuracy | % | % |
| Training Time | X mins | Y mins |
| Trainable Params | N | M |

## 7. Conclusion & Recommendations
- Summarize which model performed better and why (usually Transfer Learning due to pre-learned features).
- Mention the successful Bonus Deployment using Streamlit, demonstrating real-world viability for the "Inqaz" application.
