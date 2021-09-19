import torch.nn as nn

import logging_handler
logger = logging_handler.get_logger(__name__)


class VerificationNet(nn.Module):
    """
    Siamese model architecture for verification setting.
    Includes given base embedding network.
    """
    def __init__(self, embedding_net):
        super(VerificationNet, self).__init__()
        self._embedding_net = embedding_net
        self._relu = nn.ReLU()

    def __str__(self):
        return self.__class__.__name__

    def forward_one(self, x):
        output = self._embedding_net(x)
        output = output.view(output.size()[0], -1)
        output = self._relu(output)
        return output

    def forward(self, x1, x2):
        output1 = self.forward_one(x1)
        output2 = self.forward_one(x2)
        dist = (output2 - output1).pow(2).sum(1)
        return dist

    def get_embedding(self, x):
        return self.nonlinear(self._embedding_net(x))
