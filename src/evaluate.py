import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import get_data_generators

def evaluate_model(model_path, test_ds, model_name, results_dir):
    print(f"\nEvaluating {model_name}...")
    model = tf.keras.models.load_model(model_path)
    
    y_true = []
    y_pred_probs = []
    
    for images, labels in test_ds:
        y_true.extend(labels.numpy())
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds)
        
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs).flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    # 1. Classification Report (Accuracy, Precision, Recall, F1)
    report = classification_report(y_true, y_pred, target_names=['Normal', 'Crash'])
    print(report)
    
    with open(os.path.join(results_dir, f'{model_name}_report.txt'), 'w') as f:
        f.write(report)
        
    # 2. Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Crash'], yticklabels=['Normal', 'Crash'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(results_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()
    
    # 3. ROC / AUC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_dir, f'{model_name}_roc_curve.png'))
    plt.close()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    results_dir = os.path.join(base_dir, 'results')
    models_dir = os.path.join(results_dir, 'saved_models')
    
    _, _, test_ds = get_data_generators(raw_dir)
    
    scratch_path = os.path.join(models_dir, 'scratch_cnn.h5')
    if os.path.exists(scratch_path):
        evaluate_model(scratch_path, test_ds, 'Scratch_CNN', results_dir)
        
    tl_path = os.path.join(models_dir, 'transfer_learning.h5')
    if os.path.exists(tl_path):
        evaluate_model(tl_path, test_ds, 'Transfer_Learning', results_dir)
