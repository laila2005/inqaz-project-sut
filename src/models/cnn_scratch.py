import tensorflow as tf
from tensorflow.keras import layers, models

def build_scratch_cnn(input_shape=(224, 224, 3)):
    """
    Builds a custom CNN architecture from scratch as per project requirements.
    Includes:
    - Convolutional Layers + ReLU Activation
    - MaxPooling Layers
    - Batch Normalization
    - Flatten + Dense (Fully Connected) Layers
    - Output Layer with Sigmoid (Binary Classification)
    """
    model = models.Sequential(name="Inqaz_CNN_Scratch")

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 4
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Fully Connected (Dense) Layers
    model.add(layers.Flatten())
    
    model.add(layers.Dense(512))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))  # Dropout for regularization

    model.add(layers.Dense(128))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.3))

    # Output Layer (Binary Classification: Crash vs Normal)
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

if __name__ == "__main__":
    model = build_scratch_cnn()
    model.summary()
