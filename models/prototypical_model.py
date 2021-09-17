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