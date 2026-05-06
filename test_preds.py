import os, glob, numpy as np, tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model('results/saved_models/transfer_learning.h5')

def predict_folder(folder, label, n=10):
    files = glob.glob(os.path.join(folder, '*.jpg'))[:n]
    preds = []
    for f in files:
        img = Image.open(f).convert('RGB').resize((224, 224))
        arr = (np.array(img) / 127.5) - 1.0
        arr = np.expand_dims(arr, 0)
        p = model.predict(arr, verbose=0)[0][0]
        preds.append(p)
    avg = np.mean(preds)
    print(f"\n=== Folder: {folder} ===")
    print(f"  Individual predictions: {[f'{p:.4f}' for p in preds]}")
    print(f"  Average prediction: {avg:.4f}")

predict_folder('data/raw/crash', 'crash', 10)
predict_folder('data/raw/normal', 'normal', 10)
