import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_size=1538, projection_size=256, hidden_size=128):
        super().__init__()

        self.projection = nn.Linear(input_size, projection_size)

        self.lstm = nn.LSTM(
            input_size = projection_size,
            hidden_size = hidden_size,
            bidirectional = True
        )

    def forward(self, x):
        x = self.projection(x)
        x, _ = self.lstm(x) # [W, B, 128+128] because of forward + backward
        return x
