import os
import shutil
import kagglehub

def download_and_setup_dataset():
    """
    Downloads a Car Crash dataset from Kaggle to use for the project.
    Note: Requires Kaggle account configured or kagglehub handles public datasets.
    Dataset used: https://www.kaggle.com/datasets/vishnu606/car-crash-dataset (example)
    """
    print("Downloading Car Crash Dataset...")
    
    # Download dataset via kagglehub
    # Make sure to pip install kagglehub
    # Note: Replace 'vishnu606/car-crash-dataset' with any other if it goes down
    path = kagglehub.dataset_download("vishnu606/car-crash-dataset")
    
    print(f"Dataset downloaded to: {path}")
    
    # Move to our data/raw folder
    target_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
    os.makedirs(target_dir, exist_ok=True)
    
    print("Moving files to data/raw...")
    # Copying files from the downloaded cache to our project folder
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Try to infer class from directory structure
                class_name = os.path.basename(root).lower()
                if 'crash' in class_name or 'accident' in class_name:
                    dest_class = 'crash'
                else:
                    dest_class = 'normal'
                
                class_dir = os.path.join(target_dir, dest_class)
                os.makedirs(class_dir, exist_ok=True)
                
                src_path = os.path.join(root, file)
                dest_path = os.path.join(class_dir, file)
                shutil.copy2(src_path, dest_path)
                
    print(f"Dataset successfully set up in {target_dir}!")

if __name__ == "__main__":
    download_and_setup_dataset()
