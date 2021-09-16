import gc
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import (Tuple, Union, Dict, Any)

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from helpers import read_json
from verification.splits import create_splits

import logging_handler
logger = logging_handler.get_logger(__name__)


class SessionsDataset(Dataset):

    available_modes = ['train', 'test', 'validation']

    def __init__(self, data: np.ndarray, targets: np.ndarray, **kwargs):
        """
        Dataset for collecting sessions data for NN format.
        :param kwargs: {'mode', 'transform', 'target_transform'}
        """
        super(SessionsDataset, self).__init__()

        self._data = data
        self._targets = targets
        if len(self._data) != len(self._targets):
            logger.error(f"Data vectors and target vector have different length")
            logger.error(f"{len(self._data)} and {len(self._targets)}, check it and try again.")
            raise AttributeError(f"Data vectors and target vector have different length")

        self.__mode = kwargs.get('mode', 'train')
        if self.__mode not in self.available_modes:
            logger.error(f"Provided `mode` parameters should be one from available list: {self.available_modes}")
            logger.error(f"But was given: {self.__mode}")
            raise AttributeError(f"Provided `mode` parameters should be one from available list.")

        self._transform = kwargs.get('transform', None)
        self._target_transform = kwargs.get('target_transform', None)

        self.__unique_classes, self.__n_classes = self.__get_classes()
        self._cls2id, self._id2cls = self.__index_classes(inverse=kwargs.get('inverse', True))

    def __get_classes(self) -> Tuple[np.ndarray, int]:
        """
        Select unique classes from data.
        """
        unique_classes = np.unique(self._targets)
        logger.info(f"Dataset: found {len(unique_classes)} classes in data.")
        return unique_classes, len(unique_classes)

    def __index_classes(self, inverse: bool = True) -> Union[Tuple[Dict[str, int], Dict[int, str]],
                                                             Dict[str, int]]:
        """
        Map string classes names to indexes.
        :param inverse: whether to return inverse mapping - index2class_name
        :return: mapping - class_name2index
        """
        inds = defaultdict()
        for i, cls in enumerate(self.__unique_classes):
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


class PrototypicalBatchSampler(Sampler):
    """
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.
    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    """
    available_modes = ['train', 'test', 'validation']

    def __init__(self, labels, **kwargs):
        """
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        """
        # super(PrototypicalBatchSampler, self).__init__()

        self.__labels = labels  # len(labels) == len(all_dataset) !
        self.__mode = kwargs.get('mode', 'train')
        if self.__mode not in self.available_modes:
            logger.error(f"Provided `mode` parameters should be one from available list: {self.available_modes}")
            logger.error(f"But was given: {self.__mode}")
            raise AttributeError(f"Provided `mode` parameters should be one from available list.")

        self._classes_per_it = kwargs.get("classes_per_it", None)  # n-shot
        self._sample_per_class = kwargs.get("num_samples", None)  # k-way
        self._iterations = kwargs.get("iterations", 100)

        self._unique_classes, self._classes_counts = np.unique(self.__labels, return_counts=True)  # in sorted order
        self._unique_classes = torch.LongTensor(self._unique_classes)

        # Create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c

        self._dataset_indexes = np.empty((len(self._unique_classes), max(self._classes_counts)), dtype=int) * np.nan
        self._dataset_indexes = torch.Tensor(self._dataset_indexes)

        # Count each class occurrence - store the number of samples for each class/row
        self._num_elem_per_class = torch.zeros_like(self._unique_classes)
        for idx, label in enumerate(self.__labels):
            label_idx = np.argwhere(self._unique_classes == label).item()
            self._dataset_indexes[label_idx, np.where(np.isnan(self._dataset_indexes[label_idx]))[0][0]] = idx
            self._num_elem_per_class[label_idx] += 1

    def __iter__(self):
        """
        Yield a batch of indexes of samples from data.
        """
        for iteration in range(self._iterations):
            logger.debug(f"Prototypical sampler iteration #{iteration}")
            batch_size = self._sample_per_class * self._classes_per_it
            batch = torch.LongTensor(batch_size)

            # Select classes_per_it random classes for iteration
            iter_classes_idxs = torch.randperm(len(self._unique_classes))[:self._classes_per_it]

            for i, c in enumerate(self._unique_classes[iter_classes_idxs]):
                s = slice(i * self._sample_per_class, (i + 1) * self._sample_per_class)  # create slice
                # Get indexes of labels with current class
                label_idx = torch.arange(len(self._unique_classes)).long()[self._unique_classes == c].item()
                # Get sample_per_class random data samples that belongs to current class
                samples_indexes = torch.randperm(self._num_elem_per_class[label_idx])[:self._sample_per_class]
                if len(samples_indexes) < self._sample_per_class:
                    samples_indexes = random.choices(np.arange(self._num_elem_per_class[label_idx]),
                                                     k=self._sample_per_class)
                batch[s] = self._dataset_indexes[label_idx][samples_indexes]

            # Shuffle batch
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self) -> int:
        """
        returns the number of iterations (episodes) per epoch
        """
        return self._iterations


def init_dataloader(data: np.ndarray, targets: np.ndarray,
                    mode: str, classes_per_it: int,
                    iterations: int, num_query: int, num_support: int,
                    transform=None, target_transform=None):

    return DataLoader(SessionsDataset(data, targets, mode=mode,
                                      transform=transform, target_transform=target_transform),
                      batch_sampler=PrototypicalBatchSampler(labels=targets,
                                                             mode=mode,
                                                             classes_per_it=classes_per_it,
                                                             num_samples=(num_support + num_query),
                                                             iterations=iterations))


def create_training_dataloaders(data: pd.DataFrame,
                                splitting_params_fn: str,
                                batching_params_fn: str):
    """
    Creates train/val/test dataloaders for Pytorch model training and evaluation.
    :param data: dataframe with generated features
    :param splitting_params: file with kwargs for splitting function ()
    :param batching_params_fn: file with kwargs for Prototypical Network batching
    :return: dict of dataloaders (and label encoder)
    """
    if not Path(splitting_params_fn).exists():
        logger.error(f"File with settings for splitting data was not found with path provided.")
        raise FileNotFoundError(f"File with settings for splitting data was not found with path provided.")

    if not Path(batching_params_fn).exists():
        logger.error(f"File with settings for batching data was not found with path provided.")
        raise FileNotFoundError(f"File with settings for batching data was not found with path provided.")

    splitting_params = dict(read_json(splitting_params_fn)).get("splitting_params", {})
    logger.debug(f"Splitting parameters: {splitting_params}")

    batching_params = dict(read_json(batching_params_fn)).get("batching_options", {})
    logger.debug(f"Batching parameters: {batching_params}")

    if splitting_params.get('encode_target', False):
        splits, encoder = create_splits(data, **splitting_params)
    else:
        splits = create_splits(data, **splitting_params)

    dataloaders = defaultdict()
    for ds_type, splitted_data in splits.items():
        dataloaders[ds_type] = init_dataloader(*splitted_data, mode=ds_type,
                                               classes_per_it=batching_params.get("classes_per_it_train"),
                                               iterations=batching_params.get("iterations"),
                                               num_query=batching_params.get("num_query_train"),
                                               num_support=batching_params.get("num_support_train"))
        logger.info(f"Data loader of type: {ds_type} created.")
    del splits
    _ = gc.collect()

    if splitting_params.get('encode_target', False):
        return dataloaders, encoder
    else:
        return dataloaders


