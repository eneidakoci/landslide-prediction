import pandas as pd
import numpy as np

# Load the dataset
file_path = r"C:\Users\User\Desktop\cleaned_training_dataset.csv"
df = pd.read_csv(file_path)

# Function to round to nearest 100
def round_nearest_100(x):
    return round(x / 100) * 100

# Apply rounding only if the column exists and is numeric
if 'Precipitation' in df.columns:
    df['Precipitation'] = pd.to_numeric(df['Precipitation'], errors='coerce')
    df = df.dropna(subset=['Precipitation'])
    df['Precipitation'] = df['Precipitation'].apply(round_nearest_100)
else:
    print(" 'Precipitation' column not found in the dataset.")

# Save the updated dataset
output_path = r"C:\Users\User\Desktop\cleaned_training_dataset_rounded.csv"
df.to_csv(output_path, index=False)

print(f" Precipitation column rounded to nearest 100 and saved to {output_path}")
