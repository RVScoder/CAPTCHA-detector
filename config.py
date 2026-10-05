import torch
import os
import numpy as np
import pandas as pd
from pathlib import Path

DATASET_PATH = r"C:\Users\Ratcha Chandrashekar\Desktop\Coding\Projects\CAPTCHA-detector\captcha-dataset-2.zip\captcha-unified"
IMAGE_HEIGHT = 48
BATCH_SIZE = 128
best_model_path = ""
ckpt_model_path = ""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

meta_df = pd.read_csv(os.path.join(DATASET_PATH, "metadata.csv"))
train_df = pd.read_csv(os.path.join(DATASET_PATH, "train.csv"))
test_df = pd.read_csv(os.path.join(DATASET_PATH, "test.csv"))
val_df = pd.read_csv(os.path.join(DATASET_PATH, "val.csv"))

train_df = train_df.drop(columns=["original_filename", "original_path", "extension", "sha256"])
test_df = test_df.drop(columns=["original_filename", "original_path", "extension", "sha256"])
val_df = val_df.drop(columns=["original_filename", "original_path", "extension", "sha256"])