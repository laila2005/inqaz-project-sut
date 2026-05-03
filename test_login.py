import os
os.environ["KAGGLE_API_TOKEN"] = "KGAT_1e12565547a8e8129aaaa4862b56ea05"
import kagglehub

try:
    kagglehub.login()
    print("Login successful")
    path = kagglehub.dataset_download("vishnu606/car-crash-dataset")
    print(f"Dataset downloaded to {path}")
except Exception as e:
    print(f"Error: {e}")
