import pandas as pd
import numpy as np
from typing import (NoReturn, Union, Dict, Any, Tuple, List)

from helpers import read_json
from config import init_config, config
import create_training_dataset
from eyemovements.classification import run_eyemovements_classification
from generate_features import FeatureGenerator
from verification.dataloaders import (create_training_dataloaders, create_verification_dataloader,
                                      create_selfverify_dataloader)
from verification.train_utils import (Trainer, init_model, evaluate, aggregate_SP_predictions)
from visualization import visualize_quality

class VerificationStand:

    def __init__(self, config_path: str):
        self._config_path = config_path
        init_config(config_path)
        self._model= None
        self._fgen = FeatureGenerator()
        self._trainer = Trainer()


    def run(self, mode: str) -> Union[NoReturn, Dict[str, Any]]:
        """
        Entry point for running model.
        :param mode: mode of run - 'train' or 'run'
        :return:
        """
        assert mode in ['train', 'run']
        if mode == 'train':
            self._run_train()
        else:
            self._run_verification()


    def _run_train(self) -> NoReturn:
        """
        Training of model.
        :return: -
        """
        # Creating dataset
        dataset = create_training_dataset.TrainDataset(config.get('DataPaths', 'train_data'))
        for user in dataset._users:
            print(user)
        data = dataset.create_dataset()
        del dataset

        # Make eye movements classification
        data = run_eyemovements_classification(data, is_train=True, do_estimate_quality=True)

        # Extract features
        data = self._fgen.extract_features(data, is_train=True, rescale=True)

        # Create splits for training model
        dataloaders = create_training_dataloaders(data, 20, {"features_cols": self._fgen._feature_columns,
                                                             "target_col":  self._fgen._target_column,
                                                             "encode_target":  False,
                                                             "val_split_ratio": 0.3,
                                                             "estim_quality": True,
                                                             "test_split_ratio": 0.2})
        # Run training
        self._model = self._trainer.fit(train_loader=dataloaders.get('train'),
                                        val_loader=dataloaders.get('val'),
                                        model_dir_to_save="../models_checkpoints",
                                        model_filename="model_test2.pt")

        # Test quality
        _ = evaluate(self._model, dataloader=dataloaders.get('test'),
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
        dataset = create_training_dataset.RunDataset(owner_path=config.get('DataPaths', 'owner_data'),
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
        predictions = evaluate(self._model, dataloader, estim_quality=True,
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
        predictions = evaluate(self._model, dataloader, estim_quality=estimate_quality,
                               threshold=moves_threshold, print_metrics=False, binarize=False)
        return aggregate_SP_predictions(predictions, session_threshold, policy=policy)





if __name__ == "__main__":
    stand = VerificationStand(".\\set_locations.ini")
    stand.run("run")