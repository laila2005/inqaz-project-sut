import kagglehub
import sys

try:
    path = kagglehub.dataset_download("mdfahimbinamin/car-crash-dataset")
    print(f"Success: {path}")
except Exception as e:
    print(f"Error: {e}")
