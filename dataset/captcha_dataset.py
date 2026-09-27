from PIL import Image
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from config import DATASET_PATH, IMAGE_HEIGHT, train_df, test_df, val_df
from .vocab import char2idx, idx2char, BLANK_IDX



class CaptchaDataset(Dataset):

    def __init__(self, df, dataset_path, char2idx, image_height=48):
        self.df = df
        self.dataset_path = dataset_path
        self.char2idx = char2idx
        self.image_height = image_height

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        #image
        image_path = os.path.join(self.dataset_path, row["image_path"])
        image = Image.open(image_path).convert("RGB")
        
        original_width, original_height = image.size

        new_height = self.image_height
        new_width = round((original_width * new_height) / original_height)

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS) # LANCOZ is an image interpolation method

        image = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.permute(2, 0, 1)

        #label
        label = row["label"]

        encoded_label = torch.tensor([self.char2idx[c] for c in label], dtype=torch.long)

        return image_tensor, encoded_label, label, new_width



# creating all datasets

train_dataset = CaptchaDataset(
    train_df,
    DATASET_PATH,
    char2idx,
    IMAGE_HEIGHT
)

test_dataset = CaptchaDataset(
    test_df,
    DATASET_PATH,
    char2idx,
    IMAGE_HEIGHT
)

val_dataset = CaptchaDataset(
    val_df,
    DATASET_PATH,
    char2idx,
    IMAGE_HEIGHT
)