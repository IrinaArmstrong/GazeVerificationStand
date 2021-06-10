import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from collections import namedtuple, defaultdict
from typing import (List, Dict, Any, Tuple, Optional)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import (DataLoader, Dataset, TensorDataset)

import logging_handler
logger = logging_handler.get_logger(__name__)


#---------------------------- Verification ----------------------------

Example = namedtuple("Example", ["guid", "data", "label"])
ContrastiveExample = namedtuple("ContrastiveExample", ["example_0", "example_1", "label"])


class ContrastiveDataset(Dataset):

    def __init__(self, examples: List[ContrastiveExample]):
        self.size = len(examples)
        self.examples = examples

    def __getitem__(self, index: int):
        return (torch.from_numpy(self.examples[index].example_0).float(),
                torch.from_numpy(self.examples[index].example_1).float(),
                self.examples[index].label)

    def __len__(self):
        return self.size



def create_contrastive_dataset(data: List[Example], batch_size: int) -> DataLoader:
    """
    Creates max-margin Contrastive loss function,
    takes a pair of embedding vectors x0 and x1 as inputs.
    It has a margin parameter m > 0 to impose a lower bound
    on the distance between a pair of samples with different labels.
    """
    pairs = []
    label2vec = defaultdict(list)
    # Collect by label
    for example in data:
        label2vec[example.label].append(example)

    for example in data:
        x0 = example
        x1 = None
        # Positive example from the same label: label == 1
        while x1 is None or x1.guid == x0.guid:
            x1 = random.choice(label2vec[x0.label])
        pairs.append(ContrastiveExample(example_0=x0.data,
                                        example_1=x1.data,
                                        label=1))

        # Negative example from different label: label == 0
        x1 = None
        while x1 is None or x1.label == x0.label:
            x1 = random.choice(data)
        pairs.append(ContrastiveExample(example_0=x0.data,
                                        example_1=x1.data,
                                        label=0))
    return DataLoader(ContrastiveDataset(pairs), batch_size=batch_size, num_workers=0)


def wrap_dataset(data: np.ndarray, labels: np.ndarray) -> List[Example]:
    """
    Create dataset for a sequence of [(label, data), (label, data)...] samples.
    """
    return [Example(guid=i, data=d, label=l) for i, (d, l)
            in tqdm(enumerate(zip(data, labels)))]


# --------------------------- Training -----------------------------


def train_val_split(df: pd.DataFrame, session_id_col: str,
                    target_col: str, val_ratio: float):
    train_sess_ids, val_sess_ids = train_test_split(df[session_id_col].unique(),
                                                    test_size=val_ratio, random_state=11,
                                                    stratify=df.groupby(by=session_id_col)[target_col].apply(lambda x:
                                                                                                              x.unique()[
                                                                                                                  0]).values)
    train_df = df.loc[df[session_id_col].isin(train_sess_ids)]
    val_df = df.loc[df[session_id_col].isin(val_sess_ids)]
    logger.info(f"Train dataset of {train_df[target_col].nunique()} users with shape: {train_df.shape}")
    logger.info(f"Validation dataset of {val_df[target_col].nunique()} users with shape: {val_df.shape}")
    return train_df, val_df


def train_test_split_by_user(df: pd.DataFrame,
                             target_col: str, n_users_test: int):
    unseen_users = set()
    while len(unseen_users) < n_users_test:
        unseen_users.add(random.choice(df[target_col].unique(), k=n_users_test))
    unseen_users = list(unseen_users)
    test_df = df.loc[df[target_col].isin(unseen_users)]
    train_df = df.loc[~df[target_col].isin(unseen_users)]
    logger.info(f"Train dataset of {train_df[target_col].nunique()} users with shape: {train_df.shape}")
    logger.info(f"Test dataset of {test_df[target_col].nunique()} users with shape: {test_df.shape}")
    return train_df, test_df


def create_splits(data: pd.DataFrame, session_id_col: str,
                  data_col: str, target_col: str, encode_target: bool,
                  val_split_ratio: float,
                  estim_quality: bool=False,
                  n_users_test: int=None):
    """
    Creates train/val splits.
    :return: data splits (and label encoder for target).
    """
    splits = {}
    if encode_target:
        le = LabelEncoder().fit(data[target_col].values)
        data[target_col] = le.transform(data[target_col].values)
        logger.info(f"Totally classes encoded: {len(list(le.classes_))}")

    if estim_quality and n_users_test:
        data, test_data = train_test_split_by_user(data, target_col=target_col, n_users_test=n_users_test)
        splits['test'] = (test_data[data_col], test_data[target_col])

    train_data, val_data = train_val_split(data, session_id_col=session_id_col, val_ratio=val_split_ratio)
    splits['train'] = (train_data[data_col], train_data[target_col])
    splits['val'] = (val_data[data_col], val_data[target_col])
    if encode_target:
        return splits, le
    else:
        return splits


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

    def __init__(self, labels, classes_per_it, num_samples, iterations):
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

    datasets = defaultdict()
    for ds_type, splitted_data in splits.items():
        datasets[ds_type] = init_dataloader(*splitted_data, mode=ds_type,
                                            classes_per_it=batching_params.get("classes_per_it_train"),
                                            iterations=batching_params.get("iterations"),
                                            num_query=batching_params.get("num_query_train"),
                                            num_support=batching_params.get("num_support_train"))
        logger.info(f"Dataloader of type: {ds_type} created")
    del splits
    if splitting_params.get('encode_target', False):
        return datasets, encoder
    else:
        return datasets


def create_embeddings_dataloader(data: pd.DataFrame,
                                 batch_size: int,
                                 features_cols: List[str],
                                 target_col: str):
    """
    Creates single dataloader for Pytorch model running.
    :param data: dataframe with generated features
    :param batch_size: size of single batch
    :return: dataloader
    """
    dataset = TensorDataset(torch.from_numpy(data[features_cols].values.astype(np.float32)).float(),
                            torch.from_numpy(data[target_col].values).float())
    dataloader = DataLoader(dataset, batch_size, num_workers=0)
    return dataloader


def create_verification_dataloader(data_samples_1: pd.DataFrame,
                                   data_samples_2: pd.DataFrame,
                                   feature_columns: List[str],
                                   target_col: str=None,
                                   max_samples: int = 200,
                                   batch_size: int=10) -> DataLoader:
    """
    Create dataset and dataloader for run mode of verification.
    :param data_samples_1: owner data
    :param data_samples_2: others data (labeled or not)
    :param feature_columns: feature columns to select from data
    :param target_col: column of target value (if presented in data)
    :param max_samples: maximum number of samples to create
    :return: -
    """
    # if pd.notnull(target_col):


    ds1_inds = list(data_samples_1.index)
    ds2_inds = list(data_samples_2.index)
    pairs = []

    if len(ds1_inds) * len(ds2_inds) > max_samples:
        # Take only max_samples pairs
        for i, pair in enumerate(product(ds1_inds, ds2_inds)):
            if i < max_samples:
                ex0 = data_samples_1.iloc[pair[0]]
                ex1 = data_samples_2.iloc[pair[1]]
                pairs.append(ContrastiveExample(example_0=ex0[feature_columns].values.astype(np.float32),
                                                example_1=ex1[feature_columns].values.astype(np.float32),
                                                label=1 if target_col and (ex0[target_col] == ex1[target_col]) else 0))
            else:
                return DataLoader(ContrastiveDataset(pairs), batch_size=batch_size, num_workers=0)
    else:
        # Take them all
        inds_pairs = list(product(ds1_inds, ds2_inds))
        for pair in inds_pairs:
            ex0 = data_samples_1.iloc[pair[0]]
            ex1 = data_samples_2.iloc[pair[1]]
            pairs.append(ContrastiveExample(example_0=ex0[feature_columns].values.astype(np.float32),
                                            example_1=ex1[feature_columns].values.astype(np.float32),
                                            label=1 if target_col and (ex0[target_col] == ex1[target_col]) else 0))
        return DataLoader(ContrastiveDataset(pairs), batch_size=batch_size, num_workers=0)


def create_selfverify_dataloader(data_samples: pd.DataFrame,
                                 feature_columns: List[str],
                                 max_samples: int = 200,
                                 batch_size: int=10) -> DataLoader:
    """
    Create dataset and dataloader for run mode of verification.
    :param data_samples_1: owner data
    :param data_samples_2: others data (labeled or not)
    :param feature_columns: feature columns to select from data
    :param target_col: column of target value (if presented in data)
    :param max_samples: maximum number of samples to create
    :return: -
    """
    ds_inds = list(data_samples.index)
    pairs = []

    # Take them all
    inds_pairs = [pair for pair in list(product(ds_inds, ds_inds)) if pair[0] != pair[1]]
    for pair in inds_pairs:
        ex0 = data_samples.iloc[pair[0]]
        ex1 = data_samples.iloc[pair[1]]
        pairs.append(ContrastiveExample(example_0=ex0[feature_columns].values.astype(np.float32),
                                        example_1=ex1[feature_columns].values.astype(np.float32),
                                        label=1))
    return DataLoader(ContrastiveDataset(pairs), batch_size=batch_size, num_workers=0)