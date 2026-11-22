from pathlib import Path
import pandas as pd

DATA_FILE = Path("data/UNSW_NB15_training-set.xlsx")

print("=" * 50)
print("Loading Dataset...")
print("=" * 50)

df = pd.read_excel(DATA_FILE)

print("\nDataset Loaded Successfully!")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nAttack Categories:")
print(df["attack_cat"].value_counts())

print("\nLabel Distribution:")
print(df["label"].value_counts())

print("\nFirst 5 Rows:")
print(df.head())