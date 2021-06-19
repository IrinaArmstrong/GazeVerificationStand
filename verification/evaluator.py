# Basic
import os
import sys
sys.path.insert(0, "..")
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

import torch
import torch.nn as nn

from config import config
from helpers import read_json
from verification.model import EmbeddingNet, PrototypeNet
from verification.train_utils import (seed_everything, copy_data_to_device, init_model)
import logging_handler
logger = logging_handler.get_logger(__name__)


class PrototypicalEvaluator:

    def __init__(self, embedding_model: nn.Module=None, to_load: bool=True):
        self._eval_parameters  = dict(read_json(config.get("GazeVerification", "verification_params")))
        self._device = torch.device(self._eval_parameters.get("training_options", {}).get("device", "cpu"))
        self.__init_prototypical_model(embedding_model, to_load)
        self.__init_parameters()
        self.__modes = ['identification', 'verification', 'embeddings', 'run']
        seed_everything(seed_value=11)

    def __init_parameters(self):
        self._num_users_per_it = self._eval_parameters.get("identification_setting", {}).get("num_users_per_it", 1)
        self._support_samples_per_user = self._eval_parameters.get("identification_setting",
                                                                   {}).get("support_samples_per_user", 1)
        self._query_samples_per_user = self._eval_parameters.get("identification_setting",
                                                                 {}).get("query_samples_per_user", 1)


    def __init_prototypical_model(self, embedding_model: nn.Module, to_load: bool):
        """
        Create Prototypical model form given Embedding base model or loaded from file.
        :param embedding_model: pre-trained model;
        :param to_load: whether to load pre-trained weights from file;
        """
        self._model_parameters = dict(read_json(config.get("GazeVerification", "model_params")))
        if to_load:
            fname = os.path.join(sys.path[0],
                                 self._eval_parameters.get("pretrained_model_location", "."),
                                 self._eval_parameters.get("model_name", "model.pt"))
            if not os.path.isfile(fname):
                logger.error(f"No pretrained model file found in given path: {fname}.\n",
                             f"Check path and retry.")
                return
            else:
                logger.info(f"Loading model from: {fname}")
                embedding_model = init_model(EmbeddingNet, parameters=self._model_parameters,
                                          dir=self._eval_parameters.get("pretrained_model_location", "."),
                                          filename=self._eval_parameters.get("model_name", "model.pt"))
        self._protypical_model = PrototypeNet(embedding_model)
        logger.info(f"Prototypical model created.")

    def _select_threshold_sp(self):
        """
        Select threshold of verification setting on validation data with given metric.
        :return:
        """
        # todo: write!

    def _select_support(self, df: pd.DataFrame,
                        users: List[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select support samples from data for each user
        and form 3d matrix: [users*support_samples_per_user, n_channels, n_features] and target vector.
        It is assumed that support_samples_per_user parameter is the same for each user.
        :return: data array and targets array.
        """
        support_data = []
        support_targets = []
        # If not selected any users, then take all of them
        if not users:
            users = df['user_id'].unqiue()

        for user, user_df in df.loc[df['user_id'].isin(users)].groupby(by='user_id'):
            user_df = user_df.sample(n=self._support_samples_per_user, replace=True)
            support_data.extend(user_df['data_scaled'].to_list())
            support_targets.extend([user] * self._support_samples_per_user)
        return (np.vstack(support_data).reshape(len(users) * self._support_samples_per_user, 2, -1, order='F'),
                np.array(support_targets))


    def _split_support_query(self, df: pd.DataFrame,
                        users: List[int] = None) -> Tuple[Tuple[np.ndarray, np.ndarray], pd.DataFrame]:
        """
        Split data to support and query samples for each user.
        For support form 3d matrix: [users*support_samples_per_user, n_channels, n_features] and target vector.
        For query create dataframe with all data queries.
        :return: support data and targets, query DF.
        """
        support_data = []
        support_targets = []
        query_dfs = []

        # If not selected any users, then take all of them
        if not users:
            users = df['user_id'].unqiue()

        for user, user_df in df.loc[df['user_id'].isin(users)].groupby(by='user_id'):
            # Select support
            supp_user_df = user_df.sample(n=self._support_samples_per_user, replace=True)
            support_data.extend(supp_user_df['data_scaled'].to_list())
            support_targets.extend([user] * self._support_samples_per_user)
            # Select queries
            if self._query_samples_per_user != 1:
                query_user_df = user_df.loc[~user_df.index.isin(supp_user_df.index)].sample(n=self._query_samples_per_user,
                                                                                            replace=True)
            else:
                query_user_df = user_df.loc[~user_df.index.isin(supp_user_df.index)]
            query_dfs.append(query_user_df)

        support_data = np.vstack(support_data).reshape(len(users) * self._support_samples_per_user, 2, -1, order='F')
        support_targets = np.array(support_targets)

        return (support_data, support_targets), pd.concat(query_dfs)

    def evaluate(self, mode: str, kwargs: Dict[str, Any]):
        assert (mode in self.__modes), f"Mode should be one of available: {self.__modes}."
        logger.error(f"Mode should be one of available: {self.__modes}. Given '{mode}'.")
        if mode == 'identification':
            self.__eval_identification(**kwargs)
        elif mode == 'verification':
            self.__eval_verification(**kwargs)
        elif mode == 'embeddings':
            self.__eval_embeddings(**kwargs)
        else:
            self.__run_verification(**kwargs)


    def __eval_identification(self, data: pd.DataFrame):
        pass


    def __eval_verification(self, data: pd.DataFrame):
        pass


    def __eval_embeddings(self, data: pd.DataFrame):
        pass


    def __run_verification(self, data: pd.DataFrame):
        pass

