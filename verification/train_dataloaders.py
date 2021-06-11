import random
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader, Dataset

from verification.splits import create_splits

import logging_handler
logger = logging_handler.get_logger(__name__)


# --------------------------- Training -----------------------------

class SessionsDataset(Dataset):

    def __init__(self, data: np.ndarray, targets: np.ndarray,
                 mode: str = 'train', transform=None, target_transform=None):
        super(SessionsDataset, self).__init__()
        self._data = data
        self._targets = targets
        self._mode = mode
        self._transform = transform
        self._target_transform = target_transform

        self._unique_classes, self._n_classes = self._get_classes(self._targets)
        self._cls2id, self._id2cls = self._index_classes(self._unique_classes)

    def _get_classes(self, classes: np.ndarray):
        unique_classes = np.unique(classes)
        print(f" ---- Dataset: Found {len(unique_classes)} classes ---- ")
        return unique_classes, len(unique_classes)

    def _index_classes(self, unique_classes: np.ndarray, inverse: bool = True):
        inds = defaultdict()
        for i, cls in enumerate(unique_classes):
            inds[cls] = i
        if inverse:
            inv_inds = {v: k for k, v in inds.items()}
            return dict(inds), inv_inds
        return inds

    def __getitem__(self, idx):
        x = self._data[idx]
        y = self._targets[idx]
        if self._transform:
            x = self._transform(x)
        if self._target_transform:
            y = self._target_transform(y)
        return x, y

    def __len__(self):
        return len(self._data)


def init_dataset(data, targets, mode, transform=None, target_transform=None):
    dataset = SessionsDataset(data, targets, mode=mode,
                              transform=transform, target_transform=target_transform)
    return dataset


class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.
    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, mode: str, classes_per_it, num_samples, iterations):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.mode = mode
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True) # in sorted order
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        '''
        Yield a batch of indexes of samples from data.
        '''
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]  # select cpi random classes for iteration
            last_ind = 0
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)  # create slice
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                if len(sample_idxs) < spc:
                    sample_idxs = random.choices(np.arange(self.numel_per_class[label_idx]), k=spc)
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations


def init_sampler(labels: np.ndarray, mode: str, classes_per_it: int,
                 iterations: int, num_query: int, num_support: int):
    num_samples = num_support + num_query

    return PrototypicalBatchSampler(labels=labels,
                                    mode=mode,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=iterations)


def init_dataloader(data, targets, mode: str, classes_per_it: int,
                    iterations: int, num_query: int, num_support: int):
    dataset = init_dataset(data, targets, mode)
    sampler = init_sampler(dataset._targets, mode,
                           classes_per_it=classes_per_it,
                           iterations=iterations,
                           num_query=num_query,
                           num_support=num_support)

    dataloader = DataLoader(dataset, batch_sampler=sampler)
    return dataloader


def create_training_dataloaders(data: pd.DataFrame, splitting_params: Dict[str, Any],
                                batching_params: Dict[str, Any]):
    """
    Creates train/val/test dataloaders for Pytorch model training and evaluation.
    :param data: dataframe with generated features
    :param splitting_params: kwargs for splitting function
    :param batching_params: kwargs for Prototypical Network batching
    :return: dict of dataloaders (and label encoder)
    """

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
        logger.info(f"Dataloader of type: {ds_type} created")
    del splits
    if splitting_params.get('encode_target', False):
        return dataloaders, encoder
    else:
        return dataloaders


