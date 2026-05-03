import pandas as pd

df = pd.read_excel(r"G:\car crash\archive\dataset_database.xlsx")
print(df.head())
print(df.columns)
print(df['image_type'].value_counts() if 'image_type' in df.columns else "no image_type")
