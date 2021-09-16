import random
import numpy as np
import pandas as pd
from typing import Tuple, Union, Dict, Any
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import logging_handler
logger = logging_handler.get_logger(__name__)


def k_way_train_test_split(df: pd.DataFrame,
                           target_col: str, k_classes_test: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data to train and test parts for k-way classification task.
    Thus, some classes (k) will only be in the test dataset and
    none of their samples will appear in the training data.
    :return: train and test datasets as pd.DataFrames
    """
    # Select new, unseen classes in amount of `n_users_test`
    unseen_classes = set()
    while len(unseen_classes) < k_classes_test:
        unseen_classes.add(random.choice(df[target_col].unique(), k=k_classes_test))
    unseen_classes = list(unseen_classes)
    logger.info(f"To test data comes {k_classes_test} classes: {unseen_classes}")

    # Set data with those classes as test
    test_df = df.loc[df[target_col].isin(unseen_classes)]
    train_df = df.loc[~df[target_col].isin(unseen_classes)]

    logger.info(f"Train dataset of {train_df[target_col].nunique()} classes with shape: {train_df.shape}")
    logger.info(f"Test dataset of {test_df[target_col].nunique()} classes with shape: {test_df.shape}")

    return train_df, test_df


def train_val_split(df: pd.DataFrame, session_id_col: str,
                    target_col: str, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split full train data to train and validation parts with given ratio (val_ratio).
    :return: train and validation datasets as pd.DataFrames
    """
    target_stratification = df.groupby(by=session_id_col)[target_col].apply(lambda x: x.unique()[0]).values
    train_sess_ids, val_sess_ids = train_test_split(df[session_id_col].unique(),
                                                    test_size=val_ratio, random_state=11,
                                                    stratify=target_stratification)
    train_df = df.loc[df[session_id_col].isin(train_sess_ids)]
    val_df = df.loc[df[session_id_col].isin(val_sess_ids)]

    logger.info(f"Train dataset of {train_df[target_col].nunique()} users with shape: {train_df.shape}")
    logger.info(f"Validation dataset of {val_df[target_col].nunique()} users with shape: {val_df.shape}")

    return train_df, val_df


def create_splits(data: pd.DataFrame, session_id_col: str,
                  data_col: str, target_col: str, encode_target: bool,
                  val_split_ratio: float,
                  estim_quality: bool = False,
                  n_users_test: int = None) -> Union[Dict[str, Tuple[np.ndarray, np.ndarray]],
                                                     Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Any]]:
    """
    Creates train/validation (and if chosen - test) splits.
    :return: data splits (and label encoder for target).
    """
    splits = {}
    if encode_target:
        le = LabelEncoder().fit(data[target_col].values)
        data[target_col] = le.transform(data[target_col].values)
        logger.info(f"Totally classes encoded: {len(list(le.classes_))}")

    if estim_quality and n_users_test:
        data, test_data = k_way_train_test_split(data, target_col=target_col, k_classes_test=n_users_test)
        splits['test'] = (test_data[data_col].values, test_data[target_col].values)

    train_data, val_data = train_val_split(data, session_id_col=session_id_col, target_col=target_col,
                                           val_ratio=val_split_ratio)
    splits['train'] = (train_data[data_col].values, train_data[target_col].values)
    splits['validation'] = (val_data[data_col].values, val_data[target_col].values)
    if encode_target:
        return splits, le
    else:
        return splits