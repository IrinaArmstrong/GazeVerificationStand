import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Callable, Dict, AnyStr, Union, TypeVar, NewType

from eyemovements.eyemovements_utils import (get_movement_indexes, GazeState,
                                             get_amplitude_and_angle, get_path_and_centroid)

class Metric(ABC):

    def __init__(self, metric_name: str):
        self._name = metric_name

    @abstractmethod
    def estimate(self, data: Union[List[np.ndarray], np.ndarray], **kwargs) -> Any:
        pass


Metric = NewType("Metric", Metric)


class AverageSaccadesNumber(Metric):
    """
    Average number of saccades.
    """

    def __init__(self):
        super().__init__("AverageSaccadesNumber")

    def estimate(self, data: List[np.ndarray], **kwargs) -> Any:
        """
        Estimates on Dataset - list of sessions.
        """
        return np.mean([len(get_movement_indexes(session, GazeState.saccade)) for session in data])


class AverageSaccadesAmplitude(Metric):
    """
    Average amplitude of saccades.
    """

    def __init__(self):
        super().__init__("AverageSaccadesAmplitude")

    def estimate(self, data: List[np.ndarray], **kwargs) -> Any:
        """
        Estimates on Dataset - list of sessions.
        kwargs: 'gaze' - gaze x and y coordinates (as 2 dim. array)
        """
        return np.mean([get_amplitude_and_angle(gaze[sac])[0]
                        for movements, gaze in zip(data, kwargs.get('gaze'))
                        for sac in get_movement_indexes(movements, GazeState.saccade)])

