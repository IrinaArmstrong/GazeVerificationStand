import torch
import torch.nn as nn

from models.cbam import CBAM
import logging_handler
logger = logging_handler.get_logger(__name__)


class EmbeddingNet(nn.Module):
    """
    Base architecture for embeddings creation.
    """
    def __init__(self, embedding_size: int = 100):
        super(EmbeddingNet, self).__init__()
        self._embedding_size = embedding_size

        self.cnn1 = nn.Sequential(
            nn.Conv1d(2, 128, kernel_size=3, stride=2),
            CBAM(gate_channels=128, reduction_ratio=16, no_spatial=True),
            nn.MaxPool1d(5, stride=2),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 256, kernel_size=3, stride=2),
            CBAM(gate_channels=256, reduction_ratio=16, no_spatial=True),
            nn.MaxPool1d(3, stride=2),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(128, self._embedding_size))

    def __str__(self):
        return self.__class__.__name__

    def forward(self, x):
        if len(x.size()) == 2:
            x = torch.unsqueeze(x, 1)
        output = self.cnn1(x.float())
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output
