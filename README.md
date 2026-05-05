# Inqaz: AI Emergency Rescue System

Egypt's AI-powered emergency response application. This project uses Computer Vision and Image Processing to detect car crashes from photos and simulate automatic alerts to the MOI (122) and Ambulance (123) with GPS coordination.

## Project Structure
```text
inqaz-project-sut/
├── data/
│   └── raw/                # Raw dataset images (crash/ and normal/)
├── notebooks/
│   └── Inqaz_Pipeline.ipynb # Full pipeline walkthrough notebook
├── src/
│   ├── models/
│   │   ├── cnn_scratch.py          # Custom CNN architecture (from scratch)
│   │   └── transfer_learning.py    # MobileNetV2 Transfer Learning (manual)
│   ├── setup_dataset.py    # Dataset download and organization
│   ├── preprocess.py       # Resize, normalize [-1,1], augment, split
│   ├── train.py            # Model training (both architectures)
│   └── evaluate.py         # Metrics, confusion matrix, ROC curve
├── results/
│   └── saved_models/       # Trained .h5 model files
├── docs/
│   ├── Final_Defense_Study_Guide.md  # Viva preparation document
│   └── report_outline.md            # Report structure outline
├── app.py                  # Streamlit Web Deployment (Bonus)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Setup Instructions

1. **Install Dependencies**
   Ensure you have Python 3.9+ installed.
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**
   The dataset must contain >1,000 real images of car crashes and normal traffic.
   ```bash
   python src/setup_dataset.py
   ```

3. **Train Models**
   Runs preprocessing and trains both the Custom Scratch CNN and the Transfer Learning model.
   ```bash
   python src/train.py
   ```

4. **Evaluate Results**
   Generates Accuracy, Precision, Recall, F1-score, Confusion Matrices, and ROC curves.
   ```bash
   python src/evaluate.py
   ```

5. **Run Deployment App (Bonus)**
   Launch the Streamlit web dashboard.
   ```bash
   streamlit run app.py
   ```

## University Project Requirements Met

| Requirement | Status | Details |
|---|---|---|
| Real Dataset (>1,000 images) | ✅ | 3,000 images from Kaggle (no built-in datasets) |
| Preprocessing | ✅ | Resize 224×224, Normalize [-1,1], Augmentation, 70/15/15 Split |
| Scratch CNN | ✅ | Custom CNN built layer-by-layer (`src/models/cnn_scratch.py`) |
| Transfer Learning (Manual) | ✅ | MobileNetV2 + custom head, 2-phase training (`src/models/transfer_learning.py`) |
| Model Comparison | ✅ | Side-by-side comparison in notebook and report |
| Evaluation Metrics | ✅ | Accuracy, Precision, Recall, F1, Confusion Matrix, ROC/AUC |
| Notebook | ✅ | `notebooks/Inqaz_Pipeline.ipynb` |
| Deployment (Bonus) | ✅ | Streamlit app with simulated emergency API (`app.py`) |

## Team
**Inqaz Team** — Shorouk Academy, SUT
