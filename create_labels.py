import pandas as pd

# Load CSV
df = pd.read_csv("data/raw/Data_Entry_2017.csv")

# Create binary label: 1 if Pneumonia present, else 0
df["label"] = df["Finding Labels"].apply(lambda x: 1 if "Pneumonia" in x else 0)

# Check distribution
print("Label counts:")
print(df["label"].value_counts())
