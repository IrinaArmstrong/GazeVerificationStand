# Basic
import numpy as np
from collections import defaultdict
from typing import (Tuple, Union, Dict)
from torch.utils.data import Dataset

import logging_handler
logger = logging_handler.get_logger(__name__)


class DummyDataset(Dataset):
    def __init__(self, unique_classes: np.ndarray,
                 samples_per_class: int, n_features: int, **kwargs):
        """ Dummy dataset for debugging/testing purposes. """
        super(DummyDataset, self).__init__()

        self._samples_per_class = samples_per_class
        self._unique_classes = unique_classes
        self._data = np.random.uniform(low=0.5, high=1.3, size=(self._samples_per_class * self._unique_classes,
                                                                n_features))
        self._targets = np.hstack([np.full((self._samples_per_class,), c) for c in self._unique_classes]).reshape(-1)

        self._transform = kwargs.get('transform', None)
        self._target_transform = kwargs.get('target_transform', None)

        self._cls2id, self._id2cls = self.__index_classes(inverse=kwargs.get('inverse', True))

    def __index_classes(self, inverse: bool = True) -> Union[Tuple[Dict[str, int], Dict[int, str]],
                                                             Dict[str, int]]:
        """
        Map string classes names to indexes.
        """
        inds = defaultdict()
        for i, cls in enumerate(self._unique_classes):
            inds[cls] = i
        if inverse:
            inv_inds = {v: k for k, v in inds.items()}
            return dict(inds), inv_inds
        return inds

    def get_targets(self) -> np.ndarray:
        return self._targets

    def get_data(self) -> np.ndarray:
        return self._data

    def __getitem__(self, idx):
        """ Get item from dataset by index. """
        x = self._data[idx]
        y = self._targets[idx]
        # Transform data
        if self._transform:
            x = self._transform(x)
        # Transform target
        if self._target_transform:
            y = self._target_transform(y)
        return x, y

    def __len__(self):
        """ Length of dataset. """
        return len(self._data)

