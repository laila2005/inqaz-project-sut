import os
import shutil
import pandas as pd
from tqdm import tqdm

def organize_data(source_dir, dest_dir, excel_path):
    print("Reading labels from Excel...")
    df = pd.read_excel(excel_path)
    
    crash_dir = os.path.join(dest_dir, "crash")
    normal_dir = os.path.join(dest_dir, "normal")
    
    os.makedirs(crash_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)
    
    crash_count = 0
    normal_count = 0
    max_per_class = 1500 # We will take 1500 of each to get a nice big 3,000 image dataset

    print("Organizing images into class directories...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        img_name = row['subject']
        label = row['collision']
        
        src_path = os.path.join(source_dir, img_name)
        
        if not os.path.exists(src_path):
            continue
            
        if label == 'y' and crash_count < max_per_class:
            dst_path = os.path.join(crash_dir, img_name)
            shutil.copy2(src_path, dst_path)
            crash_count += 1
        elif label == 'n' and normal_count < max_per_class:
            dst_path = os.path.join(normal_dir, img_name)
            shutil.copy2(src_path, dst_path)
            normal_count += 1
            
        if crash_count >= max_per_class and normal_count >= max_per_class:
            break

    print(f"\nDataset construction complete!")
    print(f"Total Crash images: {crash_count}")
    print(f"Total Normal images: {normal_count}")

if __name__ == "__main__":
    SOURCE = r"G:\car crash\archive\dataset"
    EXCEL = r"G:\car crash\archive\dataset_database.xlsx"
    DEST = r"g:\image-processing project\inqaz-project-sut\data\raw"
    
    organize_data(SOURCE, DEST, EXCEL)
