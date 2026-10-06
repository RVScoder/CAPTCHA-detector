import torch
from predict import model, predict
from tts import speak_captcha

input_image = input("Enter the image path: ")

prediction, confidence, character_confidences = predict(
    input_image,
    model
)

print("Predicted CAPTCHA:", prediction)
print(f"Confidence: {confidence:.2%}")

speak_captcha(prediction)