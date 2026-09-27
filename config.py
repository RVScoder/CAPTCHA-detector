import torch
import os
import numpy as np
import pandas as pd

DATASET_PATH = ""
IMAGE_HEIGHT = 48
BATCH_SIZE = 128


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

meta_df = pd.read_csv(os.path.join(DATASET_PATH, "images", "metadata.csv"))
train_df = pd.read_csv(os.path.join(DATASET_PATH, "images", "train.csv"))
test_df = pd.read_csv(os.path.join(DATASET_PATH, "images", "test.csv"))
val_df = pd.read_csv(os.path.join(DATASET_PATH, "images", "val.csv"))

train_df = train_df.drop(columns=["original_filename", "original_path", "extension", "sha256"])
test_df = test_df.drop(columns=["original_filename", "original_path", "extension", "sha256"])
val_df = val_df.drop(columns=["original_filename", "original_path", "extension", "sha256"])