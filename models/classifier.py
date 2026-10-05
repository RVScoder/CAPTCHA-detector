import torch
import torch.nn as nn
import torch.nn.functional as F

class CharacterClassifier(nn.Module):
    def __init__(self, input_size=512, num_classes=63):
        super().__init__()

        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x
