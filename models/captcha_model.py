import torch
import torch.nn as nn
from .cnn import CNNEncoder
from .bilstm import BiLSTM
from .classifier import CharacterClassifier

class CAPTCHARecognizer(nn.Module):
    def __init__(self, cnn, bilstm, classifier):
        super().__init__()
        
        self.cnn = cnn
        self.bilstm = bilstm
        self.classifier = classifier

    def forward(self, x):
        x = self.cnn(x)

        # reshaping the output from cnn in order to feed the BiLSTM
        B, C, H, W = x.shape
        x = x.permute(3, 0, 1, 2)
        x = x.reshape(W, B, C*H)

        x = self.bilstm(x)
        x = self.classifier(x)

        return x

cnn_encoder = CNNEncoder()
bilstm_encoder = BiLSTM()
character_classifier = CharacterClassifier()

model = CAPTCHARecognizer(
    cnn = cnn_encoder,
    bilstm = bilstm_encoder,
    classifier = character_classifier
)