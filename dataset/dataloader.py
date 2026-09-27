from config import BATCH_SIZE
from .captcha_dataset import train_dataset, test_dataset, val_dataset
from .collate_fn import captcha_collate_fn
from torch.utils.data import DataLoader

# creating dataloaders for all datasets

train_loader = DataLoader(
    train_dataset,
    collate_fn = captcha_collate_fn,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 4,
    pin_memory = True
)

test_loader = DataLoader(
    test_dataset,
    collate_fn = captcha_collate_fn,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 4,
    pin_memory = True
)

val_loader = DataLoader(
    val_dataset,
    collate_fn = captcha_collate_fn,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 4,
    pin_memory = True
)