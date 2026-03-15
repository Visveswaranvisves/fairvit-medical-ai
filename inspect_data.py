import pandas as pd

# Load the CSV file
df = pd.read_csv("data/raw/Data_Entry_2017.csv")

# Show first 5 rows
print("First 5 rows:")
print(df.head())

# Show all column names
print("\nColumn names:")
print(df.columns)
