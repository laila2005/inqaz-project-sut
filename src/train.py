import os
import tensorflow as tf
import matplotlib.pyplot as plt
from preprocess import get_data_generators
from models.cnn_scratch import build_scratch_cnn
from models.transfer_learning import build_transfer_learning_model, unfreeze_for_finetuning

def plot_history(history, model_name, save_dir):
    """Saves accuracy and loss curves."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{model_name} - Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{model_name}_accuracy.png'))
    plt.close()
    
    # Loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} - Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{model_name}_loss.png'))
    plt.close()

def train_models():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    results_dir = os.path.join(base_dir, 'results')
    models_dir = os.path.join(base_dir, 'results', 'saved_models')
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading data...")
    train_ds, val_ds, test_ds = get_data_generators(raw_dir)
    
    epochs = 15
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]

    # --- 1. Train Scratch CNN ---
    print("\n--- Training Scratch CNN ---")
    scratch_model = build_scratch_cnn()
    scratch_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    history_scratch = scratch_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    plot_history(history_scratch, 'Scratch_CNN', results_dir)
    scratch_model.save(os.path.join(models_dir, 'scratch_cnn.h5'))

    # --- 2. Train Transfer Learning Model ---
    print("\n--- Training Transfer Learning Model (Phase 1) ---")
    tl_model, base_model = build_transfer_learning_model()
    tl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    history_tl_phase1 = tl_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs // 2,
        callbacks=callbacks
    )
    
    print("\n--- Fine-tuning Transfer Learning Model (Phase 2) ---")
    tl_model = unfreeze_for_finetuning(tl_model, base_model, num_layers_to_unfreeze=20)
    history_tl_phase2 = tl_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs // 2,
        callbacks=callbacks
    )
    
    # We combine history for simplicity or just plot phase 2
    plot_history(history_tl_phase2, 'Transfer_Learning_Finetuned', results_dir)
    tl_model.save(os.path.join(models_dir, 'transfer_learning.h5'))

    print("\nTraining Complete! Models and plots saved.")

if __name__ == "__main__":
    train_models()
