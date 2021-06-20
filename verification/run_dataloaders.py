import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from collections import namedtuple, defaultdict
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

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


def create_embeddings_dataloader(data: np.ndarray, targets: np.ndarray,
                                 batch_size: int):
    """
    Creates single dataloader for Pytorch model running.
    """
    return DataLoader(TensorDataset(torch.from_numpy(data.astype(np.float32)).float(),
                            torch.from_numpy(targets).float()), batch_size, num_workers=0)


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
