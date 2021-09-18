import sys
import torch
import numpy as np
from typing import Tuple
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

import logging_handler
logger = logging_handler.get_logger(__name__)


class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs: torch.Tensor, target: torch.Tensor,
                 loss: torch.Tensor, loss_type: str, epoch_num: int):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    @classmethod
    def name(cls):
        raise NotImplementedError


class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model.
    Log accuracy metric during training.
    """

    def __init__(self):
        super(AccumulatedAccuracyMetric, self).__init__()
        # Train
        self._train_correct = 0
        self._train_total = 0
        # Validation
        self._val_correct = 0
        self._val_total = 0

    def __call__(self, outputs: torch.Tensor, target: torch.Tensor,
                 loss: torch.Tensor, loss_type: str, epoch_num: int) -> Tuple[float, float]:
        # Get max from probas
        pred = torch.IntTensor([1 if p > 0.5 else 0 for p in outputs[0].data.numpy()])
        if loss_type == "train":
            self._train_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            self._train_total += target.size(0)
        elif "val" in loss_type:  # can be different names: `val`, `validation` etc.
            self._val_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            self._val_total += target.size(0)
        else:
            logger.error(f"Unknown loss type: {loss_type}")
        return self.value()

    def reset(self):
        # Train
        self._train_correct = 0
        self._train_total = 0
        # Validation
        self._val_correct = 0
        self._val_total = 0

    def value(self) -> Tuple[float, float]:
        # adding epsilon to prevent zero-division is some cases
        return (100 * float(self._train_correct) / (self._train_total + sys.float_info.epsilon),
                100 * float(self._val_correct) / (self._val_total + sys.float_info.epsilon))

    @classmethod
    def name(cls) -> str:
        return 'Accuracy'


class LossCallback(Metric):
    """
    Works with all models.
    Collect loss items during training and returns them to analysis.
    """
    def __init__(self):
        super(LossCallback, self).__init__()
        self._train_losses = defaultdict(list)
        self._val_losses = defaultdict(list)

    def __call__(self, outputs: torch.Tensor, target: torch.Tensor,
                 loss: torch.Tensor, loss_type: str, epoch_num: int) -> Tuple[float, float]:
        if loss_type == "train":
            self._train_losses[epoch_num].append(loss)
        elif "val" in loss_type:  # can be different names: `val`, `validation` etc.
            self._val_losses[epoch_num].append(loss)
        else:
            logger.error(f"Unknown loss type: {loss_type}")
        return self.value()

    def reset(self):
        pass

    def value(self) -> Tuple[float, float]:
        return (np.mean(self._train_losses.get(max(self._train_losses.keys()))),
                np.mean(self._val_losses.get(max(self._val_losses.keys()))))

    @classmethod
    def name(cls) -> str:
        return 'Loss'


class TensorboardCallback(Metric):
    """
    Works with all models.
    Collect loss items during training and write them to Tensorboard.
    """
    def __init__(self, log_dir: str):
        super(TensorboardCallback, self).__init__()
        self._writer = SummaryWriter(log_dir=log_dir)

    def __call__(self, outputs: torch.Tensor, target: torch.Tensor,
                 loss: torch.Tensor, loss_type: str, epoch_num: int) -> str:
        if loss_type == "train":
            self._writer.add_scalar('loss/train', loss, epoch_num)
        elif "val" in loss_type:  # can be different names: `val`, `validation` etc.
            self._writer.add_scalar('loss/val', loss, epoch_num)
        else:
            logger.error(f"Unknown loss type: {loss_type}")
            self._writer.add_scalar('loss/unknown', loss, epoch_num)
        return self.value()

    def reset(self):
        pass

    def value(self) -> str:
        return "Tensorboard running"

    @classmethod
    def name(cls) -> str:
        return 'TensorboardCallback'
