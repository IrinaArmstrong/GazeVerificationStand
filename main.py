import pandas as pd
import numpy as np
from typing import (NoReturn, Union, Dict, Any, Tuple, List)

import datasets
from helpers import read_json
from config import init_config, config
from verification.trainer import Trainer
from data_utilities import restructure_gaze_data, normalize_gaze
from eyemovements.eyemovements_classifier import EyemovementsClassifier
from verification.train_dataloaders import create_training_dataloaders
from verification.run_dataloaders import create_verification_dataloader

from verification.train_utils import init_model, evaluate_verification
from visualizations.visualization import visualize_quality

import logging_handler
logger = logging_handler.get_logger(__name__)


class VerificationStand:

    def __init__(self, config_path: str):
        self._config_path = config_path
        init_config(config_path)
        self.__available_modes = ['train', 'run']
        self._model = None
        self._trainer = Trainer()

    def run(self, mode: str):
        """
        Entry point for running model.
        :param mode: mode of run - 'train' or 'run'
        :return:
        """
        if mode not in self.__available_modes:
            logger.error(f"Selected mode was not recognized. Choose one from available: {self.__available_modes}")
        if mode == 'train':
            logger.info(f"Running `train` mode of stand...")
            self._run_train()
        else:
            logger.info(f"Running `verification` (`run`) mode of stand...")
            self._run_verification()

    def _run_train(self):
        """
        Training of model.
        """
        # Creating dataset
        dataset = datasets.TrainDataset(config.get('DataPaths', 'train_data'))
        data = dataset.create_dataset()
        del dataset

        # Make eye movements classification
        data = run_eyemovements_classification(data, is_train=True, do_estimate_quality=True)

        # Pre-process and normalize gaze
        data = restructure_gaze_data(data, is_train=True, params_path=config.get('Preprocessing', 'processing_params'))
        data = normalize_gaze(data, to_restore=False, to_save=True,
                              checkpoint_dir=config.get('GazeVerification', 'pretrained_model_location'))

        # Create splits for training model
        dataloaders = create_training_dataloaders(data, splitting_params_fn=config.get('Preprocessing',
                                                                           'processing_params'),
                                                  batching_params_fn=config.get('GazeVerification', 'model_params'))

        # Run training
        self._model = self._trainer.fit(train_loader=dataloaders.get('train'),
                                        val_loader=dataloaders.get('val'))

        # Test quality
        _ = evaluate_verification(self._model, dataloader=dataloaders.get('test'),
                                  estim_quality=True, threshold=0.55)


    def _run_verification(self):

        verification_params = dict(read_json(config.get('GazeVerification', 'verification_params')))
        estimate_quality = bool(verification_params.get("estimate_quality", 0))

        # Initialize model
        if self._model is None:
            self._model = init_model(filename=config.get('GazeVerification', 'pretrained_model_fn'))
        else:
            print(self._model)

        # Creating dataset
        dataset = datasets.RunDataset(owner_path=config.get('DataPaths', 'owner_data'),
                                      others_path=config.get('DataPaths', 'run_data'),
                                      estimate_quality=estimate_quality)
        print("\nOwner:")
        self.__owner = dataset._owner
        print(dataset._owner)
        print("\nOthers users:")
        for user in dataset._others:
            print(user)

        owner_data = dataset.get_owner_data()
        others_data = dataset.get_others_data()
        if estimate_quality:
            others_data_targets = others_data.groupby(by=['session_id']).agg({'session_target':
                                                                                  lambda x: np.unique(x)[0],
                                                                              "filename":
                                                                                  lambda x: np.unique(x)[0]})
            others_data_targets = others_data_targets.to_dict('records')

        # Make eye movements classification and extract features
        owner_data = run_eyemovements_classification(owner_data, is_train=True, do_estimate_quality=True)
        owner_data = self._fgen.extract_features(owner_data, is_train=True, rescale=True)

        others_data = run_eyemovements_classification(others_data, is_train=True, do_estimate_quality=True)
        others_data = self._fgen.extract_features(others_data, is_train=True, rescale=True)

        print(f"Owner data: {owner_data.shape}")
        print(f"Others data: {others_data.shape}")

        self_threshold = self.__create_threshold(owner_data,
                                                 moves_threshold=verification_params.get("moves_threshold", 0.6),
                                                 default_threshold=verification_params.get("session_threshold", 0.5),
                                                 policy=verification_params.get("policy"))

        verification_results = {}
        for id, session in others_data.groupby(by='session_id'):
            session = session.reset_index(drop=True)
            (result, proba) = self.__evaluate_session(owner_data, session, estimate_quality=estimate_quality,
                                                      moves_threshold=verification_params.get("moves_threshold", 0.6),
                                                      session_threshold=self_threshold,
                                                      policy=verification_params.get("policy"))
            verification_results[id] = (result, proba)
        if estimate_quality:
            self.__print_results(self_threshold, verification_results, others_data_targets)
            self.__estimate_quality(verification_results, others_data_targets)
        else:
            self.__print_results(self_threshold, verification_results)

        return verification_results

    def __print_results(self, self_threshold: float,
                        results: Dict[int, Tuple[int, float]], true_targets: List[Dict[str, int]]=[]):
        print(f"With defined self verification threshold: {self_threshold}")
        print(f"Results are:")
        if len(true_targets):
            for idx, (prediction, probability) in results.items():
                fn = true_targets[idx].get('filename', '').split("\\")[-1]
                print(f"\nSession: {fn}")
                print(f"{idx}: {prediction} predicted with proba {probability}, "
                      f"true is: {true_targets[idx].get('session_target', None)}")
        else:
            for idx, (prediction, probability) in results.items():
                print(f"{idx}: {prediction} predicted with proba {probability}")


    def __estimate_quality(self, results: Dict[int, Tuple[int, float]],
                           true_targets: List[Dict[str, int]] = []):
        y_true = []
        y_pred = []
        y_pred_probas = []
        for idx, (prediction, probability) in results.items():
            y_true.append(true_targets[idx].get('session_target', None))
            y_pred.append(prediction)
            y_pred_probas.append(probability)
        visualize_quality(y_true, y_pred, y_pred_probas)


    def __create_threshold(self, owner_data: pd.DataFrame,
                           moves_threshold: float, default_threshold: float,
                           policy: str='mean'):
        dataloader = create_selfverify_dataloader(owner_data,
                                                  feature_columns=self._fgen._feature_columns)
        predictions = evaluate_verification(self._model, dataloader, estim_quality=True,
                                            threshold=moves_threshold, print_metrics=False,
                                            binarize=False)
        if policy == 'mean':
            return np.mean(predictions)
        elif policy == "max":
            return np.max(predictions)
        elif policy.startswith('quantile'):
            q = float(policy.split("_")[-1])
            return np.quantile(predictions, q=q, interpolation='higher')
        else:
            print("Specified incorrect predictions aggregation policy, returns default value")
            return default_threshold


    def __evaluate_session(self, owner_data: pd.DataFrame,
                           others_data: pd.DataFrame,
                           estimate_quality: bool, moves_threshold: float,
                           session_threshold: float, policy: str='mean') -> Tuple[int, float]:
        dataloader = create_verification_dataloader(owner_data, others_data,
                                                    feature_columns=self._fgen._feature_columns,
                                                    target_col=self._fgen._target_column)
        predictions = evaluate_verification(self._model, dataloader, estim_quality=estimate_quality,
                                            threshold=moves_threshold, print_metrics=False, binarize=False)
        return aggregate_SP_predictions(predictions, session_threshold, policy=policy)


if __name__ == "__main__":
    logger = logging_handler.get_logger(__name__)
    stand = VerificationStand(".\\set_locations.ini")
    stand.run("run")