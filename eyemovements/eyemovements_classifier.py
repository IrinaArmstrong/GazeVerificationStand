# Basic
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import (List)
from pathlib import Path

from config import init_config, config

import logging_handler
logger = logging_handler.get_logger(__name__)

import warnings
warnings.filterwarnings('ignore')

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
                 config_path: str='set_locations.ini'):

        implemented_algorithms = ['ivdt']
        available_modes = ['calibrate', 'run']

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

        # If config is not pre-initialized
        if len(config.sections()) == 0:
            # Read config and init config here
            if Path(config_path).exists():
                init_config(config_path)
            else:
                logger.error(f"No pre-initialized config given and no configuration file found at {config_path}.")
                raise FileNotFoundError


    def __init_algorithm(self):
        """
        Creates instance of selected algorithm with given parameters.
        :return: algorithm class object.
        """
        pass


    def classify_eyemovements(self, data: pd.DataFrame,
                              sp_only: bool=True,
                              estimate: bool=True,
                              visualize: bool=True) -> pd.DataFrame:
        """
        Make eye movements classification in training or running mode.
        :param data: dataframe with gaze data
        :param sp_only: whether return only SP eye movents dataset
        :param estimate: estimate quality of classification with special metrics
        :param visualize: to output visualizations
        :return: dataframe with SP moves only.
        """
        pass