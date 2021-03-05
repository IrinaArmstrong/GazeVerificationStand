import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from collections import namedtuple, defaultdict
from typing import (List, Dict, Any)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import (DataLoader, Dataset)


#---------------------------- PREPROCESSING UTILITIES ----------------------------

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



def create_splits(data: pd.DataFrame, features_cols: List[str],
                  target_col: str, encode_target: bool,
                  val_split_ratio: float,
                  estim_quality: bool=False,
                  test_split_ratio: float=None):
    """
    Creates train/val/test splits.
    :param data: dataframe with generated features
    :param features_cols: features column names
    :param target_col: target column name
    :param encode_target: whether to encode target to numeric type (and return encoder)
    :param val_split_ratio: validation data split (0 < ratio < 1)
    :param estim_quality: whether to make prediction on test data (and create test dataset)
    :param test_split_ratio: test data split (0 < ratio < 1)
    :return: data splits (and label encoder for target).
    """
    splits = defaultdict()

    if encode_target:
        le = LabelEncoder().fit(data[target_col].values)
        data[target_col] = le.transform(data[target_col].values)
        print(f"Totally classes encoded: {len(list(le.classes_))}")

    if estim_quality and test_split_ratio:
        (train_data, test_data, train_targ, test_targ) = train_test_split(data[features_cols].values,
                                                                          data[target_col].astype(int).values.reshape(-1),
                                                                          test_size=test_split_ratio,
                                                                          random_state=11,
                                                                          stratify=data[target_col].astype(int).values.reshape(-1))
        (test_data, val_data, test_targ, val_targ) = train_test_split(test_data,
                                                                      test_targ,
                                                                      test_size=val_split_ratio,
                                                                      random_state=11,
                                                                      stratify=test_targ)
        print(f"Data shapes train: {train_data.shape}, validation: {val_data.shape}, test: {test_data.shape}")
        splits['train'] = (train_data, train_targ)
        splits['val'] = (val_data, val_targ)
        splits['test'] = (test_data, test_targ)
    else:
        (train_data, val_data, train_targ, val_targ) = train_test_split(data[features_cols].values,
                                                                        data[target_col].astype(int).values.reshape(-1),
                                                                        test_size=val_split_ratio,
                                                                        random_state=11,
                                                                        stratify=data[target_col].astype(int).values.reshape(-1))
        print(f"Data shapes train: {train_data.shape}, validation: {val_data.shape}")
        splits['train'] = (train_data, train_targ)
        splits['val'] = (val_data, val_targ)
    if encode_target:
        return splits, le
    else:
        return splits



def create_training_dataloaders(data: pd.DataFrame, batch_size: int,
                                splitting_params: Dict[str, Any]):
    """
    Creates train/val/test dataloaders for Pytorch model training and evaluation.
    :param data: dataframe with generated features
    :param batch_size: size of single batch
    :param splitting_params: kwargs for splitting function
    :return: dict of dataloaders (and label encoder)
    """

    if splitting_params.get('encode_target', False):
        splits, encoder = create_splits(data, **splitting_params)
    else:
        splits = create_splits(data, **splitting_params)

    datasets = defaultdict()
    for ds_type, splitted_data in splits.items():
        datasets[ds_type] = create_contrastive_dataset(wrap_dataset(*splitted_data), batch_size=batch_size)
        print(f"Dataset {ds_type} length: {len(datasets[ds_type].dataset)}")
        print(f"Classes balance:\n", pd.Series([ex.label
                                                for ex in datasets[ds_type].dataset.examples]).value_counts())
    del splits
    if splitting_params.get('encode_target', False):
        return datasets, encoder
    else:
        return datasets


def create_dataloader(data: pd.DataFrame, batch_size: int,
                      features_cols: List[str], target_col: str):
    """
    TODO: Добавить разделение данных владельца и тестовых, организовать сравнение - ???
    Creates single dataloader for Pytorch model running.
    :param data: dataframe with generated features
    :param batch_size: size of single batch
    :return: dataloader
    """
    dataloader = create_contrastive_dataset(wrap_dataset(data[features_cols].values,
                                                         data[target_col].astype(int).values.reshape(-1)),
                                            batch_size=batch_size)
    print(f"Dataset length: {len(dataloader.dataset)}")
    print(f"Classes balance:\n", pd.Series([ex.label for ex in dataloader.dataset.examples]).value_counts())
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
        for i, pair in enumerate(product([ds1_inds, ds2_inds])):
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