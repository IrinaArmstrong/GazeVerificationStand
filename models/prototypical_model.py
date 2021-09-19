import torch
import torch.nn as nn
import torch.nn.functional as F

from verification.loss import euclidean_dist

import logging_handler
logger = logging_handler.get_logger(__name__)


class PrototypeNet(nn.Module):
    """
    Prototypical model architecture for identification setting.
    Includes given base embedding network.
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
    """
    def __init__(self, embedding_net: nn.Module):
        super(PrototypeNet, self).__init__()
        # Embeddings network
        self._net = embedding_net
        # Runtime variables
        self._has_prototypes = False
        self._unique_classes = None
        self._prototypes = None

    def __str__(self):
        return self.__class__.__name__

    def _get_embeddings(self, batch):
        """ Just return embeddings of input vectors. """
        return self._net(batch)

    def init_prototypes(self, input_data: torch.tensor, labels: torch.tensor):
        """
        Initialize prototypes from provided data anf labels.
        """
        input_data = self._get_embeddings(input_data)
        self._unique_classes = torch.unique(labels)  # unique classes from support
        classes_indexes = [labels.eq(c).nonzero().squeeze(1) for c in
                           self._unique_classes]  # classes indexes in support
        self._prototypes = torch.stack([input_data[idxs].mean(0) for idxs in classes_indexes])
        self._has_prototypes = True

    def _get_predictions(self, input_data, return_dists: bool = False):
        """
        ???
        :param input_data:
        :param return_dists:
        """
        if not self._has_prototypes:
            logger.error("""First initialize prototypes using `init_prototypes` 
                             or directly put support set into `forward`""")
            raise AttributeError("""First initialize prototypes using `init_prototypes` 
                                    or directly put support set into `forward`""")
        # Count distances from input data to prototypes
        dists = euclidean_dist(input_data, self._prototypes)
        log_p_y = F.log_softmax(-dists, dim=1).view(input_data.size(0), len(self._unique_classes))
        pred_prototypes = torch.argmax(log_p_y, 1).numpy().astype(int)

        # Convert to classes
        pred_classes = [self._unique_classes[p] for p in pred_prototypes]
        if return_dists:
            return pred_classes, F.softmax(log_p_y)
        return pred_classes

    def forward(self, batch, return_dists: bool = False,
                return_embeddings: bool = False,
                has_support: bool = False, n_support: int = 0):
        """
        Forward pass of Prototypical network:
        - get embedding for input data
        - initialize new prototypes, optionally
        - count distances from input data to prototypes
        - select the closest one and predict appropriate class
        - return classes, (embeddings, distances, optionally)
        """
        # If input data has some vectors for creating new prototypes
        #  and last column of it contains target values
        if has_support:
            data = batch[0]
            targets = batch[-1]
            num_quiery_batch = input.size(0) - n_support
            # Get embeddings from network
            embeddings = self._get_embeddings(data)
            # Initialize prototypes
            self.init_prototypes(data, targets[:n_support])
        else:
            embeddings = self._get_embeddings(batch)

        if return_embeddings:
            return self._get_predictions(embeddings, return_dists), embeddings

        return self._get_predictions(embeddings, return_dists)