import random
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import logging_handler
logger = logging_handler.get_logger(__name__)

#---------------------------- Splitting data ----------------------------

def train_test_split_by_user(df: pd.DataFrame,
                             target_col: str, n_users_test: int):
    """
    Split all data to train and unseen test parts by users.
    Thus, some users will only be in the test dataset and
    none of their records will appear in the training data.
    :return: train and test datasets as pd.DataFrames
    """
    unseen_users = set()
    while len(unseen_users) < n_users_test:
        unseen_users.add(random.choice(df[target_col].unique(), k=n_users_test))
    unseen_users = list(unseen_users)
    test_df = df.loc[df[target_col].isin(unseen_users)]
    train_df = df.loc[~df[target_col].isin(unseen_users)]
    logger.info(f"Train dataset of {train_df[target_col].nunique()} users with shape: {train_df.shape}")
    logger.info(f"Test dataset of {test_df[target_col].nunique()} users with shape: {test_df.shape}")
    return train_df, test_df


def train_val_split(df: pd.DataFrame, session_id_col: str,
                    target_col: str, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split full train data to train and validation parts with given ratio (val_ratio).
    :return: train and validation datasets as pd.DataFrames
    """
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


def create_splits(data: pd.DataFrame, session_id_col: str,
                  data_col: str, target_col: str, encode_target: bool,
                  val_split_ratio: float,
                  estim_quality: bool=False,
                  n_users_test: int=None):
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
        data, test_data = train_test_split_by_user(data, target_col=target_col, n_users_test=n_users_test)
        splits['test'] = (test_data[data_col], test_data[target_col])

    train_data, val_data = train_val_split(data, session_id_col=session_id_col, val_ratio=val_split_ratio)
    splits['train'] = (train_data[data_col], train_data[target_col])
    splits['val'] = (val_data[data_col], val_data[target_col])
    if encode_target:
        return splits, le
    else:
        return splits