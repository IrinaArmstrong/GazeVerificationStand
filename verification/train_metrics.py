import torch
import numpy as np
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

import logging_handler
logger = logging_handler.get_logger(__name__)

# -------------------- Metrics ---------------------

class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss, loss_type, epoch_num):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class AccumulatedAccuracyMetric(Metric):
    """
    todo: split to train and validation
    Works with classification model.
    Log accuracy metric during training.
    """

    def __init__(self):
        super(AccumulatedAccuracyMetric, self).__init__()
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss, loss_type, epoch_num):
        # Get max from probas
        pred = torch.IntTensor([1 if p > 0.5 else 0 for p in outputs[0].data.numpy()])
        self.correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        self.total += target.size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'


class LossCallback(Metric):
    """
    Works with all models.
    Collect loss items during training and returns them to analysis.
    """
    def __init__(self):
        super(LossCallback, self).__init__()
        self.train_losses = defaultdict(list)
        self.val_losses = defaultdict(list)

    def __call__(self, outputs, target, loss, loss_type, epoch_num):
        if loss_type == "train":
            self.train_losses[epoch_num].append(loss)
        elif loss_type == "val":
            self.val_losses[epoch_num].append(loss)
        return self.value()

    def reset(self):
        pass

    def value(self):
        return np.mean(self.train_losses.get(max(self.train_losses.keys())))

    def name(self):
        return 'Loss'


class TensorboardCallback(Metric):
    """
    Works with all models.
    Collect loss items during training and write them to Tensorboard.
    """
    def __init__(self, log_dir: str):
        super(TensorboardCallback, self).__init__()
        self._writer = SummaryWriter(log_dir=log_dir)

    def __call__(self, outputs, target, loss: float, loss_type: str, epoch_num: int):
        if loss_type == "train":
            self._writer.add_scalar('loss/train', loss, epoch_num)
        elif loss_type == "val":
            self._writer.add_scalar('loss/val', loss, epoch_num)
        else:
            logger.warning(f"Unknown loss type: {loss_type}")
            self._writer.add_scalar('loss/unknown', loss, epoch_num)
        return self.value()

    def reset(self):
        pass

    def value(self):
        return "Tensorboard running"

    def name(self):
        return 'TensorboardCallback'