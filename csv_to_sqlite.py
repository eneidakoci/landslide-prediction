import pandas as pd
import sqlite3
from pyproj import Transformer
from pathlib import Path

transformer = Transformer.from_crs("EPSG:2199", "EPSG:4326", always_xy=True)

def gk_to_latlon(x, y):
    """
    Convert Albanian Gauss-Krüger coordinates (EPSG:2199) to lat/lon
    """
    try:
        lon, lat = transformer.transform(x, y)
        if not (39.5 <= lat <= 42.7 and 19.0 <= lon <= 21.5):
            return None, None
        return lat, lon
    except Exception:
        return None, None

csv_paths = [
    "landslides_utf8.csv",
    r"C:\Users\User\Desktop\Diploma\landslides_utf8.csv",
    Path.cwd() / "landslides_utf8.csv",
]

csv_path = None
for path in csv_paths:
    if Path(path).exists():
        csv_path = Path(path)
        break

if not csv_path:
    print("CSV file not found!")
    exit(1)

print(f" Loading CSV from: {csv_path}")

encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'windows-1252', 'cp1252']
df = None
for enc in encodings_to_try:
    try:
        df = pd.read_csv(csv_path, encoding=enc)
        print(f" Loaded {len(df)} rows with encoding: {enc}")
        break
    except Exception as e:
        print(f" Failed with encoding {enc}: {e}")

if df is None:
    print(" Could not load CSV.")
    exit(1)

print(f"\nColumns found: {list(df.columns)}")

if 'X' not in df.columns or 'Y' not in df.columns:
    print(" CSV must have 'X' and 'Y' columns!")
    exit(1)

print(f"\nSample Gauss-Krüger coordinates from CSV:")
for i in range(min(3, len(df))):
    x = df.loc[i, 'X']
    y = df.loc[i, 'Y']
    print(f"   Row {i}: X={x}, Y={y}")

df['Latitude'] = None
df['Longitude'] = None

converted = 0
failed = 0
print("\nConverting Gauss-Krüger coordinates to Lat/Lon...")

for idx, row in df.iterrows():
    x = row['X']
    y = row['Y']

    if pd.notna(x) and pd.notna(y):
        try:
            lat, lon = gk_to_latlon(float(x), float(y))
            if lat is not None and lon is not None:
                df.at[idx, 'Latitude'] = lat
                df.at[idx, 'Longitude'] = lon
                converted += 1
            else:
                failed += 1
        except Exception:
            failed += 1

print(f"\nConverted {converted}/{len(df)} coordinates successfully")
if failed > 0:
    print(f" Failed to convert {failed} coordinates")

valid_coords = df[(df['Latitude'].notna()) & (df['Longitude'].notna())]
if len(valid_coords) > 0:
    print(f"\nCoordinate ranges:")
    print(f"   Latitude:  {valid_coords['Latitude'].min():.6f} to {valid_coords['Latitude'].max():.6f}")
    print(f"   Longitude: {valid_coords['Longitude'].min():.6f} to {valid_coords['Longitude'].max():.6f}")
db_path = Path("landslides.db")
if db_path.exists():
    db_path.unlink()

with sqlite3.connect(db_path) as conn:
    df.to_sql('landslides', conn, if_exists='replace', index=False)
    conn.execute('CREATE INDEX IF NOT EXISTS idx_date ON landslides("Data e ndodhjes ")')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_coords ON landslides(Latitude, Longitude)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_gk ON landslides(X, Y)')

    cursor = conn.execute('SELECT COUNT(*) FROM landslides WHERE Latitude IS NOT NULL')
    valid_count = cursor.fetchone()[0]
    print(f"\nRecords with valid coordinates in DB: {valid_count}")

print("\nDATABASE CREATED SUCCESSFULLY!")
