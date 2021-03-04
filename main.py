import pandas as pd
from pprint import pprint
from typing import (NoReturn, Union, Dict, Any)

from helpers import read_json
from config import init_config, config
import create_training_dataset
from eyemovements.classification import run_eyemovements_classification
from generate_features import FeatureGenerator
from verification.dataloaders import (create_training_dataloaders, create_verification_dataloader)
from verification.train_utils import (Trainer, init_model, evaluate, aggregate_SP_predictions)

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
        #         ***** Eval results {} *****
        # Accuracy score: 0.8027522935779816
        # Balanced_accuracy_score: 0.8027522935779816
        #               precision    recall  f1-score   support
        #
        #            0       0.79      0.83      0.81       218
        #            1       0.82      0.78      0.80       218
        #
        #     accuracy                           0.80       436
        #    macro avg       0.80      0.80      0.80       436
        # weighted avg       0.80      0.80      0.80       436



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

        # Make eye movements classification and extract features
        owner_data = run_eyemovements_classification(owner_data, is_train=True, do_estimate_quality=True)
        owner_data = self._fgen.extract_features(owner_data, is_train=True, rescale=True)

        others_data = run_eyemovements_classification(others_data, is_train=True, do_estimate_quality=True)
        others_data = self._fgen.extract_features(others_data, is_train=True, rescale=True)

        print(f"Owner data: {owner_data.shape}")
        print(f"Others data: {others_data.shape}")

        verification_results = {}
        for id, session in others_data.groupby(by='session_id'):
            result = self.__evaluate_session(owner_data, session, estimate_quality=estimate_quality,
                                             moves_threshold=verification_params.get("moves_threshold", 0.5),
                                             session_threshold=verification_params.get("session_threshold", 0.5))
            verification_results[id] = result

        pprint(verification_results)
        return verification_results


    def __evaluate_session(self, owner_data: pd.DataFrame,
                           others_data: pd.DataFrame,
                           estimate_quality: bool,
                           moves_threshold: float,
                           session_threshold: float) -> int:
        dataloader = create_verification_dataloader(owner_data, others_data,
                                                    feature_columns=self._fgen._feature_columns,
                                                    target_col=self._fgen._target_column)
        predictions = evaluate(self._model, dataloader, estim_quality=estimate_quality,
                               threshold=moves_threshold)
        return aggregate_SP_predictions(predictions, session_threshold)




if __name__ == "__main__":
    stand = VerificationStand(".\\set_locations.ini")
    stand.run("run")