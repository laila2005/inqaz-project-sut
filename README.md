# Inqaz: AI Emergency Rescue System

Egypt's AI-powered emergency response application. This project uses Computer Vision and Image Processing to detect car crashes from photos and simulate automatic alerts to the MOI (122) and Ambulance (123) with GPS coordination.

## Project Structure
```text
/inqaz-project-sut
│── data/
│   ├── raw/            # Un-preprocessed dataset images
│   └── processed/      # (Optional) Saved processed data
│── docs/               # Project report outlines and documentation
│── src/                
│   ├── models/         # CNN Architecture definitions
│   │   ├── cnn_scratch.py
│   │   └── transfer_learning.py
│   ├── setup_dataset.py # Script to download/prepare the dataset
│   ├── preprocess.py   # Data augmentation, normalization, resizing
│   ├── train.py        # Model training and saving
│   └── evaluate.py     # Metric calculation and visualization
│── results/            # Saved `.h5` models and evaluation plots
│── app.py              # Streamlit Web Deployment (Bonus Phase)
│── requirements.txt    # Project dependencies
│── README.md           # This file
```

## Setup Instructions

1. **Install Dependencies**
   Ensure you have Python 3.9+ installed.
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**
   The dataset must contain >1,000 real images of car crashes and normal traffic. You can run the setup script to automatically pull a dataset from Kaggle to `data/raw/`:
   ```bash
   python src/setup_dataset.py
   ```

3. **Train Models**
   This will run the preprocessing pipeline and train both the Custom Scratch CNN and the manually built Transfer Learning model.
   ```bash
   python src/train.py
   ```

4. **Evaluate Results**
   Calculate Accuracy, Precision, Recall, F1-score, and generate Confusion Matrices and ROC curves.
   ```bash
   python src/evaluate.py
   ```

5. **Run Deployment App (Bonus)**
   Launch the web simulator.
   ```bash
   streamlit run app.py
   ```

## University Project Requirements Met:
- **Real Dataset:** Handled raw dataset, >1000 images, no built-in library data.
- **Preprocessing:** Resizing (224x224), Normalization [0,1], Augmentation, and Split applied properly in `preprocess.py`.
- **Scratch Model:** Full custom CNN built layer-by-layer in `cnn_scratch.py`.
- **Transfer Learning:** Explicitly coded manually with MobileNetV2 in `transfer_learning.py` (No drag-and-drop).
- **Evaluation:** Comprehensive metrics and generated plots.
- **Deployment:** Streamlit app `app.py`.
