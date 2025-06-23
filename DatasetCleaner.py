import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re

# Load dataset
df = pd.read_csv(r"C:\Users\User\Desktop\training_dataset.csv")

# Rename columns for easier access
df.columns = [col.strip().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_") for col in df.columns]


# Fix range values in 'Precipitation' column (e.g., '1600-1800' -> 1700)
def parse_precip(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, str) and re.match(r"^\d+\s*-\s*\d+$", val.strip()):
        low, high = map(int, re.split(r"\s*-\s*", val.strip()))
        return (low + high) / 2
    try:
        return float(val)
    except:
        return np.nan


df['Precipitation'] = df['Precipitation'].apply(parse_precip)

# Convert other numerical columns to numeric
numerical_cols = ['Elevation_above_sea_level_at_the_head_of_the_landslide_m',
                  'Slope_inclination_degrees', 'Seismicity_PGA']
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

numerical_cols.append('Precipitation')  # add precipitation to be imputed

# Impute missing numerical values using **median per class**
for col in numerical_cols:
    df[col] = df.groupby('Class')[col].transform(lambda x: x.fillna(x.median()))

# Handle categorical columns, and make sure 'unknown' is distinct
categorical_cols = ['Geology_of_the_mass', 'Land_classification', 'Erosion_rate', 'Moisture']
for col in categorical_cols:
    df[col] = df[col].fillna("unknown").astype(str).replace("nan", "unknown")

# Encode categorical columns with distinct 'unknown'
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

    # Ensure "unknown" is mapped to its own index
    if "unknown" in le.classes_:
        unknown_index = list(le.classes_).index("unknown")
    else:
        unknown_index = -1  # fallback

    label_encoders[col] = le

# Ensure Class is int
df['Class'] = df['Class'].astype(int)

# Save cleaned data
df.to_csv(r"C:\Users\User\Desktop\cleaned_training_dataset.csv", index=False)

print("✅ Dataset cleaned successfully.")
print("ℹ️  'unknown' is encoded separately. Missing numerical values are filled with class-based medians.")
