# Inqaz AI - Professor Discussion & Defense Prep

This document contains likely questions your professor will ask during your project defense, along with complete, concise, and professional answers.

---

### Q1: Why did you choose to use both a Custom CNN and Transfer Learning (MobileNetV2)?
**Answer:** We built a Custom CNN to demonstrate our foundational understanding of deep learning architectures, building layers (Convolution, Pooling, Batch Normalization) from scratch. We then used MobileNetV2 via Transfer Learning to compare our custom model against an industry-standard, highly optimized model. MobileNetV2 was specifically chosen because it is extremely lightweight and fast, making it ideal for real-time mobile and web deployments like our emergency dashboard.

### Q2: What exactly is "Transfer Learning" and why is it beneficial here?
**Answer:** Transfer learning is taking a model that has already been trained on a massive dataset (like ImageNet, which has millions of images) and reusing its "knowledge" for our specific problem. Instead of the AI having to learn basic shapes and edges from scratch, it already knows how to extract features. We just remove its original final layer and add a new custom classification head to teach it to specifically recognize "Crash" vs "Normal."

### Q3: How did you preprocess your dataset before training?
**Answer:** We applied three main preprocessing steps:
1. **Resizing:** All images were resized to exactly `224x224` pixels because neural networks require a fixed input size.
2. **Normalization:** We scaled the pixel values by dividing by `255`. This converts pixel values from `0-255` to a range of `0.0 to 1.0`, which helps the neural network converge faster during training.
3. **Data Augmentation:** We applied random horizontal flips, rotations (20%), and zooming (20%) to the training data. This prevents overfitting by ensuring the model doesn't just memorize the training images, but actually learns to generalize.

### Q4: How did you split your dataset?
**Answer:** We used a standard `70/15/15` split. 
- **70%** (2,100 images) was used for Training the model.
- **15%** (450 images) was used for Validation (tuning the model during training and triggering early stopping).
- **15%** (450 images) was strictly held out as Test Data to evaluate the final, unbiased accuracy.

### Q5: What loss function and final activation function did you use, and why?
**Answer:** 
- We used **Sigmoid** for the final activation function because this is a *Binary Classification* problem (Crash = 0, Normal = 1). Sigmoid perfectly outputs a probability between `0.0` and `1.0`.
- We used **Binary Crossentropy** as our loss function because it is the mathematical standard for penalizing wrong predictions in binary classification tasks.

### Q6: If your model gets around 70% accuracy, what are the architectural reasons for this, and how would you improve it in the future?
**Answer:** 70% is a solid baseline for a small dataset of 3,000 images, but we've identified two architectural bottlenecks for future improvement:
1. **Flattening vs. Pooling:** Currently, our transfer learning model uses a `Flatten()` layer which creates over 16 million parameters, causing the model to overfit. In the future, replacing this with a `GlobalAveragePooling2D()` layer will drop the parameter count drastically and boost validation accuracy.
2. **Preprocessing Ranges:** MobileNetV2 was originally trained on ImageNet using an input range of `[-1, 1]`, but we fed it `[0, 1]`. Adjusting our rescaling layer to match the `[-1, 1]` range would allow the pre-trained weights to perform much better.

### Q7: Explain your prediction logic in the application. How does the UI know it's a crash?
**Answer:** Because the raw dataset folders were named alphabetically (`crash` and `normal`), the training pipeline automatically assigned `Class 0` to Crash and `Class 1` to Normal. In our `app.py`, when a user uploads an image, the model outputs a probability. If the output probability is `< 0.5`, the system classifies it as a Crash (Class 0) and immediately triggers the automated emergency API sequence (fetching GPS, and dispatching ambulance/police). If it's `> 0.5`, it is flagged as a Normal situation.
