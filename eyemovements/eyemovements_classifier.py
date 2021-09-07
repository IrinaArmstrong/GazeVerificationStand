# Basic
import traceback
import pandas as pd
from typing import (List, Dict, Any)
from pathlib import Path

from helpers import read_json
from config import init_config, config
from eyemovements.ivdt_algorithm import IVDT
from eyemovements.filtering import sgolay_filter_dataset
from eyemovements.eyemovements_utils import get_sp_moves_dataset
from eyemovements.eyemovements_estimator import EyemovementsEstimator
from visualizations.visualization import visualize_eyemovements
from data_utilities import (horizontal_align_data, groupby_session, interpolate_sessions)
from eyemovements.eyemovements_metrics import all_metrics_list

import logging_handler
logger = logging_handler.get_logger(__name__)

import warnings
warnings.filterwarnings('ignore')

implemented_algorithms = ['ivdt']
available_modes = ['calibrate', 'run']

class EyemovementsClassifier:
    """
    Main class of eye movements classification module.
    It provides an opportunity to select `mode` of classification:
        - calibrate - to select preferable thresholds for algorithms on small sample of data.
                    Now, it is made by printing specific metrics (which?) and create visualizations as output.
                    In such a way - hand-made estimation;
        - run - to start classification on full available data;
    Also it provides a choice of algorithm for classification:

    """

    def __init__(self,  mode: str, algorithm: str='ivdt',
                 config_path: str='..\set_locations.ini'):

        if mode not in available_modes:
            logger.error(f"""Eye movements Classifier mode should be one from: {available_modes}.
                         Given type {mode} is unrecognized.""")
            raise NotImplementedError

        if algorithm not in implemented_algorithms:
            logger.error(f"""Eye movements Classifier implements few algorithms: {implemented_algorithms}.
                         Given algorithm type {algorithm} is unrecognized.""")
            raise NotImplementedError

        self._mode = mode
        self._algorithm_name = algorithm
        self._algorithm = None
        self._model_params = {}

        self._estimator = EyemovementsEstimator([metric() for metric in all_metrics_list])

        # If config is not pre-initialized
        if len(config.sections()) == 0:
            # Read config and init config here
            if Path(config_path).exists():
                init_config(config_path)
            else:
                logger.error(f"No pre-initialized config given and no configuration file found at {config_path}.")
                raise FileNotFoundError

        self.__init_algorithm()

    def __init_algorithm(self):
        """
        Creates instance of selected algorithm with given parameters.
        :return: algorithm class object.
        """
        self._model_params = dict(read_json(config.get('EyemovementClassification', 'model_params')))
        if self._algorithm_name == 'ivdt':
            self._algorithm = IVDT(saccade_min_velocity=self._model_params.get('saccade_min_velocity'),
                                   saccade_min_duration=self._model_params.get('min_saccade_duration_threshold'),
                                   saccade_max_duration=self._model_params.get('max_saccade_duration_threshold'),
                                   window_size=self._model_params.get('window_size'),
                                   dispersion_threshold=self._model_params.get('dispersion_threshold'))
        else:
            logger.error(f"""Eye movements Classifier implements few algorithms: {implemented_algorithms}.
                                     Given algorithm type {self._algorithm_name} is unrecognized.""")
            raise NotImplementedError

    def classify_eyemovements(self, data: pd.DataFrame,
                              sp_only: bool = True, h_align: bool = True,
                              estimate: bool = True,
                              visualize: bool = True,
                              to_save: bool = True, **kwargs) -> List[pd.DataFrame]:
        """
        Make eye movements classification in training or running mode.
        :param data: dataframe with gaze data
        :param sp_only: whether return only SP eye movents dataset
        :param estimate: estimate quality of classification with special metrics
        :param visualize: to output visualizations
        :return: dataframe with SP moves only.
        """
        # Clean beaten sessions and interpolate lost values in gaze data
        data = data.copy()
        data = interpolate_sessions(data, "gaze_X", "gaze_Y")

        # Grouping long-formed dataset by sessions
        data = groupby_session(data)

        # Filtering and taking derivatives
        data = sgolay_filter_dataset(data, **dict(read_json(config.get("EyemovementClassification",
                                                                       "filtering_params"))))
        # Filtering
        thresholds_dict = {'min_saccade_duration_threshold': self._model_params.get('min_saccade_duration_threshold'),
                           'max_saccade_duration_threshold': self._model_params.get('max_saccade_duration_threshold'),
                           'min_fixation_duration_threshold': self._model_params.get('min_fixation_duration_threshold'),
                           'min_sp_duration_threshold': self._model_params.get('min_sp_duration_threshold')}

        data = self._algorithm.get_eyemovements(data,
                                                gaze_col=['filtered_X', 'filtered_Y'],
                                                time_col='timestamps',
                                                velocity_col='velocity_sqrt',
                                                thresholds=thresholds_dict)

        if estimate:
            # kwargs: {"compare_with_all_SP", `averaging_strategy`, "amplitude_coefficient"}
            metrics = self._estimator.estimate_dataset(data,
                                                       compare_with_all_SP=kwargs.get('compare_with_all_SP', True),
                                                       averaging_strategy=kwargs.get('averaging_strategy', 'macro'),
                                                       amplitude_coefficient=kwargs.get('amplitude_coefficient', 0.33))
            quality_report = self._estimator.report_quality(metrics)
            logger.info(quality_report)

        if sp_only:
            data = get_sp_moves_dataset(data)

        # Update difference using filtered gaze coordinates
        if type(data) == list:
            for d in data:
                d['x_diff'] = d["stim_X"] - d["filtered_X"]
                d['y_diff'] = d["stim_Y"] - d["filtered_Y"]
        else:
            data['x_diff'] = data["stim_X"] - data["filtered_X"]
            data['y_diff'] = data["stim_Y"] - data["filtered_Y"]

        if visualize:
            try:
                if type(data) == list:
                    for i, d in enumerate(data):
                        visualize_eyemovements(d, to_save=to_save, session_num=i)
                else:
                    visualize_eyemovements(data, to_save=to_save)
            except Exception as ex:
                logger.error(f"""Error occurred while visualizing eye movements results:
                             {traceback.print_tb(ex.__traceback__)}""")
        if h_align:
            data = horizontal_align_data(data,
                                         grouping_cols=['user_id', 'session_id', 'stimulus_type', 'move_id'],
                                         aligning_cols=['x_diff', 'y_diff']).reset_index().rename({"index": "sp_id"},
                                                                                                  axis=1)

        return data


    def update_algorithm_parameters(self, updating_params: Dict[str, Any]):
        """
        Update algorithm parameters and thresholds in configuration file.
        :param updating_params:
        :return:
        """
        # todo: implement
        pass
