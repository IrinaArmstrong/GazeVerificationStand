import torch
from collections import defaultdict

class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss, epoch_num):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model.
    Log accuracy metric during training.
    """

    def __init__(self):
        super(AccumulatedAccuracyMetric, self).__init__()
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss, epoch_num):
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
        self.losses = defaultdict(list)

    def __call__(self, outputs, target, loss, epoch_num):
        self.losses[epoch_num].append(loss)
        return self.value()

    def reset(self):
        pass

    def value(self):
        return self.losses.get(max(self.losses.keys()))[0]

    def name(self):
        return 'Loss'