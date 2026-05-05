import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def build_transfer_learning_model(input_shape=(224, 224, 3)):
    """
    Builds a Transfer Learning model manually (no drag and drop).
    - Uses MobileNetV2 as the backbone without the top classification head.
    - Adds an explicitly coded custom head.
    """
    
    # Load pre-trained backbone
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    
    # Freeze the base layers for the first training phase
    base_model.trainable = False

    # Create new model explicitly
    model = models.Sequential(name="Inqaz_Transfer_Learning")
    
    # Add the base model
    model.add(base_model)
    
    # Add custom head explicitly in code
    model.add(layers.GlobalAveragePooling2D())
    
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(128))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.3))
    
    # Output Layer
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model, base_model

def unfreeze_for_finetuning(model, base_model, num_layers_to_unfreeze=20):
    """
    Unfreezes the top layers of the base_model for fine-tuning.
    Call this after the initial training phase of the custom head.
    """
    base_model.trainable = True
    
    # Re-freeze all layers except the last 'num_layers_to_unfreeze'
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
        
    # We must recompile the model for these changes to take effect
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5), # Use a lower learning rate for fine-tuning
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    model, base_model = build_transfer_learning_model()
    model.summary()
