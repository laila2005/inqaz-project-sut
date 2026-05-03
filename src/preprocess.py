import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory

def get_data_generators(data_dir, batch_size=32, img_size=(224, 224)):
    """
    Creates training, validation, and testing datasets using TensorFlow's 
    image_dataset_from_directory and applies necessary preprocessing.
    
    Preprocessing Steps Applied:
    1. Resizing to 224x224
    2. Data Augmentation (Rotation, Flip, Zoom) for training
    3. Normalization (Scaling [0, 255] to [0, 1])
    4. Train/Val/Test Split (70/15/15)
    """
    
    # 1. Load Datasets and Resize
    # We use a seed to ensure splits don't overlap
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.3, # We take 30% for Val+Test initially
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )
    
    val_test_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )
    
    # Split the 30% val_test_ds in half -> 15% Validation, 15% Test
    val_batches = tf.data.experimental.cardinality(val_test_ds)
    val_ds = val_test_ds.take(val_batches // 2)
    test_ds = val_test_ds.skip(val_batches // 2)

    # 2. Data Augmentation (Only applied to training data)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ])

    # 3. Normalization Layer
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    # Apply augmentation and normalization
    # Map functions for performance
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
    
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)

    # Prefetching for better performance
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds

if __name__ == "__main__":
    # Test the pipeline
    base_dir = os.path.dirname(os.path.dirname(__file__))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    
    if os.path.exists(raw_dir) and len(os.listdir(raw_dir)) > 0:
        train_ds, val_ds, test_ds = get_data_generators(raw_dir)
        print("Preprocessing Pipeline Successful!")
        print(f"Train batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
        print(f"Validation batches: {tf.data.experimental.cardinality(val_ds).numpy()}")
        print(f"Test batches: {tf.data.experimental.cardinality(test_ds).numpy()}")
    else:
        print("Raw data directory is empty or missing. Please run setup_dataset.py first.")
