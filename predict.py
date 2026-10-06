from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataset.vocab import idx2char, BLANK_IDX
from models.captcha_model import model


IMAGE_HEIGHT = 48
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model_path = r"output\best_captcha_model (4).pth"


model = model.to(device)

checkpoint = torch.load(
    best_model_path,
    map_location=device
)

model.load_state_dict(checkpoint)
model.eval()


def predict(image_path, model):

    image = Image.open(image_path).convert("RGB")

    original_width, original_height = image.size

    new_height = IMAGE_HEIGHT
    new_width = round((original_width * new_height) / original_height)

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # Convert to numpy
    image = np.array(image, dtype=np.float32) / 255.0

    image = torch.from_numpy(image)

    image = image.permute(2, 0, 1)

    image = image.unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():

        logits = model(image)

        probabilities = torch.softmax(logits, dim=2)

        predictions = probabilities.argmax(dim=2)


    # CTC decoding and confidence
    sequence = predictions[:, 0].tolist()

    decoded = []
    character_confidences = []

    previous = None

    for t, char_id in enumerate(sequence):

        if char_id == BLANK_IDX:
            previous = char_id
            continue

        if char_id == previous:
            previous = char_id
            continue

        decoded.append(char_id)

        confidence = probabilities[t, 0, char_id].item()

        character_confidences.append(confidence)

        previous = char_id


    prediction = "".join(idx2char[idx] for idx in decoded)

    # Overall confidence
    if character_confidences:
        confidence = sum(character_confidences) / len(character_confidences)
    else:
        confidence = 0.0


    return prediction, confidence, character_confidences

