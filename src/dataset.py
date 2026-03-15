import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.image_dir, row["image_path"])
        image = Image.open(img_path).convert("RGB")

        label = torch.tensor(row["label"], dtype=torch.long)
        sex = row["sex"]  # Keep as string for fairness analysis

        if self.transform:
            image = self.transform(image)

        return image, label, sex


