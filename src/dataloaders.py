import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import ChestXrayDataset


def get_dataloaders(
    train_csv,
    val_csv,
    image_dir,
    batch_size=8,
    num_workers=2
):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = ChestXrayDataset(
        dataframe=train_df,
        image_dir=image_dir,
        transform=train_transforms
    )

    val_dataset = ChestXrayDataset(
        dataframe=val_df,
        image_dir=image_dir,
        transform=val_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader
