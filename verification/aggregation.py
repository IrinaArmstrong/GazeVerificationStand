# Basic
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "..")
import warnings
warnings.filterwarnings('ignore')

from config import config
from helpers import read_json
import logging_handler
logger = logging_handler.get_logger(__name__)

class Aggregator:

    def __init__(self):
        self._agg_parameters = dict(read_json(config.get("GazeVerification",
                                                         "verification_params"))).get('aggregation_params')

    def aggregate_predictions(self, data: pd.DataFrame):
        pass


