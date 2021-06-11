import torch
import torch.nn as nn
import torch.nn.functional as F

from verification.cbam import CBAM
import logging_handler
logger = logging_handler.get_logger(__name__)

# --------------------- Models Architectures ----------------

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

    def forward(self, x):
        if len(x.size()) == 2:
            x = torch.unsqueeze(x, 1)
        output = self.cnn1(x.float())
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output



class PrototypeNet(nn.Module):
    """
    Prototypical model architecture for identification setting.
    Includes given base embedding network.
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
    """
    def __init__(self, embedding_net: nn.Module):
        super(PrototypeNet, self).__init__()
        self._net = embedding_net
        self._has_prototypes = False

    def _get_embeddings(self, batch):
        return self._net(batch)

    def init_prototypes(self, input: torch.tensor, labels: torch.tensor):
        input = self._get_embeddings(input)
        self._unique_classes = torch.unique(labels)  # unique classes from support
        self._classes_inds = [labels.eq(c).nonzero().squeeze(1) for c in
                              self._unique_classes]  # classes indexes in support
        self._prototypes = torch.stack([input[idxs].mean(0) for idxs in self._classes_inds])
        self._has_prototypes = True

    def _get_predictions(self, input, return_dists: bool = False):
        if not self._has_prototypes:
            logger.error("""First initialize prototypes using `init_prototypes` 
                             or directly put support set into `forward`""")
            raise AttributeError("""First initialize prototypes using `init_prototypes` 
                                    or directly put support set into `forward`""")
        dists = euclidean_dist(input, self._prototypes)  # count distances
        log_p_y = F.log_softmax(-dists, dim=1).view(input.size(0), len(self._unique_classes))
        pred_prototypes = torch.argmax(log_p_y, 1).numpy().astype(int)
        # Convert to classses
        pred_classes = [self._unique_classes[p] for p in pred_prototypes]
        if return_dists:
            return pred_classes, F.softmax(log_p_y)
        return pred_classes

    def forward(self, batch, return_dists: bool = False, return_embeddings: bool = False,
                has_support: bool = False, targets=None, n_support: int = 0):
        if has_support:
            data = batch[0]
            targets = batch[-1]
            num_quiery_batch = input.size(0) - n_support
            # get embeddings from network
            embeddings = self._get_embeddings(data)
            # initialize prototypes
            self.init_prototypes(data, targets[:n_support])
        else:
            embeddings = self._get_embeddings(batch)

        if return_embeddings:
            return self._get_predictions(embeddings, return_dists), embeddings

        return self._get_predictions(embeddings, return_dists)


class VerificationNet(nn.Module):
    """
    Siamese model architecture for verification setting.
    Includes given base embedding network.
    """
    def __init__(self, embedding_net):
        super(VerificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.relu = nn.ReLU()

    def forward_one(self, x):
        output = self.embedding_net(x)
        output = output.view(output.size()[0], -1)
        output = self.relu(output)
        return output

    def forward(self, x1, x2):
        output1 = self.forward_one(x1)
        output2 = self.forward_one(x2)
        dist = (output2 - output1).pow(2).sum(1)
        return dist

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))