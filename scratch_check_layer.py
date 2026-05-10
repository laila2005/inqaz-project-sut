import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
print("Last layer name:", model.layers[-1].name)
