import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from config import device, best_model_path, ckpt_model_path
from models.captcha_model import model
from dataset.dataloader import train_loader, test_loader, val_loader
from dataset.vocab import char2idx, idx2char, BLANK_IDX


criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# lr scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-5
)

import time
from difflib import SequenceMatcher

epochs = 50

best_val_loss = float("inf")
best_exact_match = 0.0

for epoch in range(epochs):

    epoch_start_time = time.time()

    # Training
    model.train()

    total_train_loss = 0.0
    total_train_samples = 0

    for batch_images, targets, label_lengths, labels, widths in train_loader:

        batch_images = batch_images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        label_lengths = label_lengths.to(device, non_blocking=True)
        input_lengths = (widths // 4).to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        logits = model(batch_images)

        # CTC expects log probabilities
        log_probs = logits.log_softmax(dim=2)

        # CTC loss
        loss = criterion(
            log_probs,
            targets,
            input_lengths,
            label_lengths
        )

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_size = batch_images.size(0)

        total_train_loss += loss.item() * batch_size
        total_train_samples += batch_size

    avg_train_loss = total_train_loss / total_train_samples

    # Validation
    model.eval()

    total_val_loss = 0.0
    total_val_samples = 0

    correct_captchas = 0
    total_captchas = 0

    correct_characters = 0
    total_characters = 0

    with torch.no_grad():

        for batch_images, targets, label_lengths, labels, widths in val_loader:

            batch_images = batch_images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            label_lengths = label_lengths.to(device, non_blocking=True)
            input_lengths = (widths // 4).to(device, non_blocking=True)

            # Forward pass
            logits = model(batch_images)

            # CTC loss
            log_probs = logits.log_softmax(dim=2)

            loss = criterion(
                log_probs,
                targets,
                input_lengths,
                label_lengths
            )

            batch_size = batch_images.size(0)

            total_val_loss += loss.item() * batch_size
            total_val_samples += batch_size

            # Greedy CTC decoding
            predictions = logits.argmax(dim=2)

            for i in range(batch_size):

                pred = predictions[:input_lengths[i], i]

                decoded = []
                previous = None

                for char_id in pred.tolist():

                    # 62 is the CTC blank
                    if char_id == 62:
                        previous = char_id
                        continue

                    # Remove consecutive duplicates
                    if char_id != previous:
                        decoded.append(char_id)

                    previous = char_id

                # Convert IDs to characters
                predicted_text = "".join(
                    idx2char[idx] for idx in decoded
                )

                actual_text = labels[i]

                # Exact match
                if predicted_text == actual_text:
                    correct_captchas += 1

                total_captchas += 1

                # Character accuracy
                matcher = SequenceMatcher(
                    None,
                    actual_text,
                    predicted_text
                )

                matching_characters = sum(
                    block.size
                    for block in matcher.get_matching_blocks()
                )

                correct_characters += matching_characters
                total_characters += len(actual_text)

    avg_val_loss = total_val_loss / total_val_samples
    exact_match_acc = correct_captchas / total_captchas
    character_acc = correct_characters / total_characters

    # Update learning rate based on validation loss
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    epoch_time = (time.time() - epoch_start_time) / 60

    print(
        f"Epoch: {epoch + 1} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Character Acc: {character_acc:.4%} | "
        f"Exact Match: {exact_match_acc:.4%} | "
        f"LR: {current_lr:.2e} | "
        f"Time: {epoch_time:.2f} mins"
    )

    # Save best model
    if exact_match_acc > best_exact_match:
        best_exact_match = exact_match_acc
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print("Best model saved.")

    # Save checkpoint
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_character_accuracy": character_acc,
            "val_exact_match_accuracy": exact_match_acc,
            "best_val_loss": best_val_loss
        },
        ckpt_model_path
    )