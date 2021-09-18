import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Any

import logging_handler
logger = logging_handler.get_logger(__name__)

# Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py


class PrototypicalLoss(nn.Module):
    """
    Loss class deriving from Module for the prototypical loss function defined below
    """
    def __init__(self, device: Union[torch.device, str]):
        super(PrototypicalLoss, self).__init__()
        self.__device = device

    def forward(self, input_data: torch.Tensor, target: torch.Tensor, n_support: int):
        return prototypical_loss(input_data, target, n_support, self.__device)


def prototypical_loss(input_data: torch.Tensor, target: torch.Tensor,
                      n_support: int, device: Union[torch.device, str]) -> Tuple[Any, Any]:
    """
    Compute the barycentres by averaging the features of _n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed and returned.
    Args:
        - n_support - number of samples to keep in account when computing
                            barycentres, for each one of the current classes.
    """
    target_cpu = target.to(device)
    input_cpu = input_data.to(device)

    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    classes = torch.unique(target_cpu)
    n_classes = len(classes)

    # assuming n_query, n_target constants for all classes
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support
    # get indexes of support samples from batch
    support_idxs = list(map(supp_idxs, classes))
    # count prototypes for each support class samples
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    # get indexes of query samples from batch
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input_data.to(device)[query_idxs]  # get embeddings of query samples
    dists = euclidean_dist(query_samples, prototypes)  # count distances
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val, acc_val


def euclidean_dist(x, y) -> torch.Tensor:
    """
    Compute euclidean distance between two tensors (x and y).
    :param x: shape N x D
    :param y: shape M x D
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)