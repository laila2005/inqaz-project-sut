<div align="center">
  <h1>🚨 Inqaz AI: Emergency Rescue System</h1>
  <p><strong>Egypt's Next-Generation AI-Powered Traffic Incident Response Platform</strong></p>
  <p>
    <a href="https://inqaz-ai.streamlit.app/" target="_blank">
      <img src="https://img.shields.io/badge/Production-Live_Deployment-success?style=for-the-badge&logo=streamlit" alt="Production Link">
    </a>
  </p>
</div>

---

## 📖 Overview

**Inqaz** (Arabic for "Rescue") is a computer vision and deep learning project designed to automate emergency response for car accidents. By analyzing images from field cameras, dashcams, or user uploads, the AI engine can instantly detect severe vehicle damage and simulate automated dispatch alerts to the Ministry of Interior (122) and Ambulance services (123) complete with precise GPS coordination.

**Production Link:** [Access the Live Web Dashboard Here](https://inqaz-ai.streamlit.app/)

## ✨ Key Features
- **Binary Image Classification:** Real-time analysis distinguishing between `Crash` and `Normal` traffic scenes.
- **Deep Learning Architecture:** Utilizes a highly optimized **MobileNetV2 Transfer Learning** model achieving 68% F1-score on entirely unseen test data.
- **Explainable AI (Grad-CAM):** Generates transparent thermal heatmaps visualizing exactly *where* the AI detected vehicle structural damage.
- **Production-Ready Web App:** A sleek, dark-mode Streamlit dashboard featuring live camera ingestion, metric tracking, and intuitive user feedback.

---

## 🚀 Getting Started

Follow these instructions to run the complete pipeline locally, from dataset ingestion to deploying the web application.

### 1. Prerequisites
Ensure you have **Python 3.9 or higher** installed on your system.

### 2. Clone and Setup Environment
Clone the repository and install the required dependencies:
```bash
# Install required Python packages
pip install -r requirements.txt
```

### 3. Data Ingestion
The system requires a robust dataset of real-world traffic scenes. Run the dataset setup script to automatically download 3,000 balanced images (1,500 Crash / 1,500 Normal) from Kaggle.
```bash
python src/setup_dataset.py
```

### 4. Model Training
Train both our Custom Baseline CNN and the advanced MobileNetV2 Transfer Learning architecture. This script handles image preprocessing (resizing, `[-1, 1]` normalization, augmentation) and model fitting.
```bash
python src/train.py
```

### 5. Evaluation
Evaluate the models on the 15% hold-out test set. This will generate the Confusion Matrices, ROC curves, and detailed Classification Reports in the `results/` folder.
```bash
python src/evaluate.py
```

### 6. Launch the Application
Start the Streamlit development server to interact with the trained AI model via the premium web interface.
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
inqaz-project-sut/
├── data/
│   └── raw/                          # Raw downloaded dataset (crash/ & normal/)
├── notebooks/
│   └── Inqaz_Pipeline.ipynb          # Jupyter notebook with full exploratory pipeline
├── src/
│   ├── models/
│   │   ├── cnn_scratch.py            # Baseline custom CNN architecture
│   │   └── transfer_learning.py      # MobileNetV2 architecture with GlobalAveragePooling2D
│   ├── setup_dataset.py              # Kagglehub automated dataset downloading
│   ├── preprocess.py                 # Data loaders, augmentation, and [-1, 1] scaling
│   ├── train.py                      # Model training execution
│   └── evaluate.py                   # Performance metrics generation
├── results/
│   └── saved_models/                 # Compiled .h5 and .keras model weights
├── docs/
│   ├── Final_Defense_Study_Guide.md  # Viva preparation and system explanation
│   ├── Final_Project_Report.md       # Comprehensive markdown report
│   ├── Final_Project_Report.pdf      # Official compiled report submission
│   ├── Inqaz_AI_Documentation.md     # High-level system architecture overview
│   ├── Inqaz_Final_Presentation.pptx # Final presentation slides
│   ├── Professor_QandA_Prep.md       # Anticipated defense Q&A
│   └── report_outline.md             # Structural outline for the documentation
├── app.py                            # Streamlit Web Deployment UI
├── requirements.txt                  # Required Python dependencies
└── README.md                         # Project documentation
```

---

## 🎓 University Requirements Checklist

| Requirement | Status | Implementation Details |
|---|:---:|---|
| **Real Dataset (>1k imgs)** | ✅ | 3,000 real-world images sourced from Kaggle. No toy datasets used. |
| **Robust Preprocessing** | ✅ | 224x224 resizing, `[-1,1]` normalization, Data Augmentation. |
| **Baseline Scratch CNN** | ✅ | Custom CNN built layer-by-layer (`src/models/cnn_scratch.py`). |
| **Transfer Learning** | ✅ | MobileNetV2 + custom classification head without `Flatten` bottlenecks. |
| **Model Evaluation** | ✅ | Accuracy, Precision, Recall, F1, Confusion Matrix, & ROC generated. |
| **Pipeline Notebook** | ✅ | Explanatory code available in `notebooks/Inqaz_Pipeline.ipynb`. |
| **Cloud Deployment** | ✅ | Full-stack deployment on Streamlit Community Cloud (`app.py`). |

---

<div align="center">
  <p>Built with ❤️ by the <b>Inqaz Team — SUT</b></p>
</div>
