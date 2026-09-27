import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_size=3072, projection_size=256, hidden_size=256):
        super().__init__()

        self.projection = nn.Linear(input_size, projection_size)

        self.lstm = nn.LSTM(
            input_size = projection_size,
            hidden_size = hidden_size,
            num_layers=2,
            dropout=0.2,
            bidirectional = True
        )

    def forward(self, x):
        x = self.projection(x)
        x, _ = self.lstm(x) # [W, B, 256+256] because of forward + backward
        return x
