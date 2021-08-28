# Basic
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import (List)

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

    def __init__(self, mode: str, algorithm: str='ivdt'):
        assert mode in ['calibrate', 'run']
        pass

    def classify_eyemovements(self, estimate: bool=True, visualize: bool=True) -> np.ndarray:
        pass