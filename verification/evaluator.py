# Basic
import os
import sys
import random
sys.path.insert(0, "..")
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, List, Dict, Any

import torch
import torch.nn as nn

from config import config
from helpers import read_json
from models.prototypical_model import EmbeddingNet, PrototypeNet
from verification.train_utils import (seed_everything, copy_data_to_device, init_model,
                                      compute_metrics_short, format_time)
from verification.run_dataloaders import create_embeddings_dataloader
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
        self._estimate_quality = bool(self._eval_parameters.get("identification_setting",
                                                                 {}).get("", False))


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
        self._protypical_model = PrototypeNet(embedding_model).eval()
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
                        users: List[int] = None) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                                          Tuple[np.ndarray, np.ndarray]]:
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

        query_dfs = pd.concat(query_dfs)
        query_data = np.vstack(query_dfs['data_scaled'].values).reshape(query_dfs.shape[0], 2, -1, order='F')
        query_targets = np.vstack(query_dfs['user_id'].values)
        # Todo: to return and save some identifier for each SP
        # query_sp_ids = np.vstack(query_dfs['splitted_sp_id'].values)
        del query_dfs

        return (support_data, support_targets), (query_data, query_targets)


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


    def __eval_identification(self, data: pd.DataFrame, users: List[int]=None, save_results: bool=True):
        if not users:
            if data['user_id'].nunique() < 2:
                logger.error(f"In identification mode in data should be more then 1 user.",
                             f"{data['user_id'].nunique()} is found.")
                raise AttributeError(f"In identification mode in data should be more then 1 user.",
                             f"{data['user_id'].nunique()} is found.")
            if self._num_users_per_it == -1:  # Take all users to identification testing
                users = data['user_id'].unique()
            else:
                users = set()
                while len(users) < self._num_users_per_it:
                    users.add(random.choice(data['user_id'].unique()))

        split = self._split_support_query(data, list(users))
        (support_data, support_targets), (queries_data, queries_targets) = split

        # Init prototypes in model
        self._protypical_model.init_prototypes(torch.FloatTensor(support_data),
                                               torch.LongTensor(support_targets))
        logger.info(f"Prototypes in model initialized for {self._protypical_model._prototypes.size(0)} classes ",
                    f"and {self._protypical_model._prototypes.size(1)} shape each.")

        # Create test dataloader with query data and targets
        identification_dataloader = create_embeddings_dataloader(queries_data, queries_targets,
                                                                 batch_size=bool(self._eval_parameters.get(
                                                                     "identification_setting", {}).get(
                                                                     "batch_size", 64)))
        predictions = self.__evaluate_prototype_identification(identification_dataloader)
        predictions = pd.DataFrame(predictions)
        if save_results:
            save_preds_fn = os.path.join(self._eval_parameters.get("output_dir", "."),
                                         f"identification_predictions_{datetime.now().strftime('%Y-%m-%d_%H:%M%S')}.csv")
            predictions.to_csv(save_preds_fn, sep=';', encoding="utf-8")
            logger.info(f"Predictions saved to: {save_preds_fn}")


    def __eval_verification(self, data: pd.DataFrame):
        pass


    def __eval_embeddings(self, data: pd.DataFrame):
        pass


    def __run_verification(self, data: pd.DataFrame):
        pass

    def __evaluate_prototype_identification(self, dataloader) -> Dict[str, List[int]]:
        """
        Identification setting on prototypes with model.
        :return: predictions for given dataset.
        """
        estim_quality = bool(self._eval_parameters.get("identification_setting",
                                                                 {}).get("estimate_quality", False))
        return_dists = bool(self._eval_parameters.get("identification_setting",
                                                       {}).get("return_distances", False))
        return_embeddings = bool(self._eval_parameters.get("identification_setting",
                                                       {}).get("return_embeddings", False))

        eval_start = datetime.now()
        self._protypical_model.eval()

        # To store predictions, true labels and so on
        pred_labels = []
        pred_dists = []
        embeddings = []

        if estim_quality:
            true_labels = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if estim_quality:
                    data = batch[:-1]
                    target = batch[-1]
                else:
                    data = batch
                    target = None

                if not type(data) in (tuple, list):
                    data = (data,)

                data = copy_data_to_device(data, self._device)
                outputs = self._protypical_model(*data, return_dists=return_dists)
                if return_dists and return_embeddings:
                    batch_pred = outputs[0]
                    dists = outputs[1].cpu().detach().numpy()
                    embs = outputs[2].cpu().detach().numpy()
                    pred_dists.extend(dists)
                    embeddings.extend(embs)
                elif return_dists:
                    batch_pred = outputs[0]
                    dists = outputs[1].cpu().detach().numpy()
                    pred_dists.extend(dists)
                elif return_embeddings:
                    batch_pred = outputs[0]
                    embs = outputs[2].cpu().detach().numpy()
                    embeddings.extend(embs)
                else:
                    outputs = outputs.cpu().detach().numpy()
                    batch_pred = outputs

                # Store labels
                if estim_quality:
                    true_labels.extend(target.tolist())
                # Store predictions
                pred_labels.extend(batch_pred)

        # Measure how long the validation run took.
        validation_time = format_time(datetime.now() - eval_start)
        if estim_quality:
            compute_metrics_short(true_labels, pred_labels)

        logger.info(
            "\tTime elapsed for evaluation: {:} with {} samples.".format(validation_time, len(dataloader.dataset)))
        outputs = {"predictions": [pred.item() for pred in pred_labels]}
        if estim_quality:
            outputs["targets"] = true_labels
        if return_dists:
            outputs['distances'] = pred_dists
        if return_embeddings:
            outputs['embeddings'] = embeddings
        return outputs

