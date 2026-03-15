import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import pandas as pd
from src.dataset import ChestXrayDataset

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Correct CSV path
csv_path = os.path.join(BASE_DIR, "data", "processed", "train_metadata.csv")

train_df = pd.read_csv(csv_path)

dataset = ChestXrayDataset(
    dataframe=train_df,
    image_dir=os.path.join(BASE_DIR, "data", "raw", "images"),
    transform=None
)

print("Dataset length:", len(dataset))
sample = dataset[0]
print("Sample output:", sample)
