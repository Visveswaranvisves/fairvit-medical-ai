import pandas as pd

df = pd.read_csv("data/processed/metadata.csv")

print("Sex distribution:")
print(df["sex"].value_counts())

print("\nLabel distribution:")
print(df["label"].value_counts())
