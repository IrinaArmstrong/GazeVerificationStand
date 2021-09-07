# Basic
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Union, Dict

import logging_handler
logger = logging_handler.get_logger(__name__)

import warnings
warnings.filterwarnings('ignore')


class GazeAnalyzer(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def _classify_eyemovements(self,
                               gaze: np.ndarray,
                               timestamps: np.ndarray,
                               velocity: np.ndarray,
                               **kwargs) -> np.ndarray:
        """
        Run classification of preprocessed gaze, velocity and time data.
        """
        pass

    @abstractmethod
    def get_eyemovements(self,
                         data: Union[pd.DataFrame, List[pd.DataFrame]],
                         gaze_col: Union[str, List[str]],
                         time_col: str,
                         velocity_col: str,
                         thresholds: Dict[str, float],
                         **kwargs) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Start full procedure of selecting eye movements from raw dataset, that can be given as:
            - single DataFrame instance
            - or list of separate DataFrames.
        Provides: preprocessing (optional), classification, merging consecutive movements and filtration.
        """
        pass


