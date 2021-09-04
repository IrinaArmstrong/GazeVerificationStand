# Basic
import traceback
import numpy as np
import pandas as pd
from typing import List, Callable, Dict, AnyStr, Union

import logging_handler
logger = logging_handler.get_logger(__name__)

from eyemovements.eyemovements_metrics import MetricType

import warnings
warnings.filterwarnings('ignore')


class EyemovementsEstimator:

    def __init__(self, metrics_list: List[MetricType]):
        self.__metrics_list = metrics_list


    def estimate_session(self, data: pd.DataFrame, **kwargs) -> Dict[AnyStr, AnyStr]:
        """
        Estimate single session quality.
        kwargs: {''}
        :return: dict with key - metric name, value - score.
        """
        pass

    def estimate_dataset(self, data: List[pd.DataFrame],
                         **kwargs) -> Union[Dict[AnyStr, AnyStr], List[Dict[AnyStr, AnyStr]]]:
        """
        Estimate single session quality.
        kwargs: {`averaging_strategy`, "amplitude_coefficient"}
        :return: dict with key - metric name, value - score.
        """
        if not (all(["stim_X" in d for d in data]) and all(["stim_Y" in d for d in data])
                and all(["gaze_X" in d for d in data]) and all(["gaze_Y" in d for d in data])
                and all(["stimulus_velocity" in d for d in data]) and all(["velocity_sqrt" in d for d in data])
                and all(["movements" in d for d in data]) and all(["timestamp" in d for d in data])):
            logger.error(f"Some necessary columns are not in data columns. Can not make estimation.")
            return {}

        metric_kwargs = {}

        # Gaze
        gaze_movements = [d['movements'].values for d in data]
        metric_kwargs['gaze_coordinates'] = [d[["gaze_X", "gaze_Y"]].values for d in data]
        metric_kwargs['gaze_velocity'] = [d['velocity_sqrt'].values for d in data]

        # Time
        metric_kwargs['timestamps'] = [d['timestamp'].values for d in data]

        # Stimulus
        if kwargs.get("compare_with_all_SP", False):
            # Special case where we compare session eye movements with preferable situation
            # with all points being part of smooth pursuits
            metric_kwargs['stimulus_eyemovements'] = [np.full_like(d['movements'].values, 3) for d in data]
        else:
            # Use gaze coordinates if special stimulus coordinated are not provided
            # (for expert mark-up of gaze data, for example, where stimulus coordinates are not necessary.)
            metric_kwargs['stimulus_eyemovements'] = [d[kwargs.get("stimulus_movements_col", 'movements')].values for d in data]

        metric_kwargs['stimulus_coordinates'] = [d[["stim_X", "stim_Y"]].values for d in data]
        metric_kwargs['stimulus_velocity'] = [d['stimulus_velocity'].values for d in data]

        # Add extra arguments from imput
        metric_kwargs.update(kwargs)
        estimates = []
        try:
            if kwargs.get("to_average", False):
                estimates = {metric.get_name(): metric.estimate(gaze_movements, **metric_kwargs)
                             for metric in self.__metrics_list}
            else:
                for gaze_movement in gaze_movements:
                    estimate_session = {metric.get_name(): metric.estimate(gaze_movement, **metric_kwargs)
                                        for metric in self.__metrics_list}
                    estimates.append(estimate_session)
        except Exception as ex:
            logger.error(f"""Error occurred while estimating eye movements results:
                                         {traceback.print_tb(ex.__traceback__)}""")
            return estimates
