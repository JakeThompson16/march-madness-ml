

from src.util.extract_cbbd import extract_cbbd_data

df = extract_cbbd_data([2025, 2024])

print(df.describe())
print(df.head())
