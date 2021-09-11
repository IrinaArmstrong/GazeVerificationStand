import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path


from config import config, init_config
from datasets import TrainDataset, RunDataset

import warnings
warnings.filterwarnings('ignore')

import logging_handler
logger = logging_handler.get_logger(__name__)


class TestDatasets(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._current_base_path = Path(__file__).parent.parent.resolve()
        init_config(str(self._current_base_path / "set_locations.ini"))

    def test_train_dataset_creation(self):
        ds = TrainDataset(ds_path=config.get('DataPaths', 'train_data'))
        logger.info(f"Found unique users: {len(ds.get_users())}")
        gaze_data = ds.create_dataset()

        self.assertTrue(len(gaze_data) > 0)
        logger.info(f"Length of dataset is: {gaze_data.shape} with columns: {gaze_data.columns}")
        logger.info(f"Size of dataset is {sys.getsizeof(gaze_data) / 1048576} Mb.")
        logger.info(f"Unique stimulus types: {gaze_data['stimulus_type'].unique()}")

    def test_run_dataset_creation(self):
        ds = RunDataset(owner_path=config.get('DataPaths', 'owner_data'),
                        others_path=config.get('DataPaths', 'run_data'),
                        estimate_quality=True)
        logger.info(f"Users in estimation dataset:")
        for user in ds.get_others():
            logger.info(user)

        logger.info(f"Verified user:")
        logger.info(ds.get_owner())

        owner_data = ds.get_owner_data()
        others_data = ds.get_others_data()

        logger.info(f"Owner data of shape: {owner_data.shape}")
        logger.info(f"Others data  of shape: {others_data.shape}")

    def test_run_dataset_creation_unknown(self):
        ds = RunDataset(owner_path=config.get('DataPaths', 'owner_data'),
                        others_path=config.get('DataPaths', 'run_data'),
                        estimate_quality=False)

        logger.info(f"Users in estimation dataset:")
        for user in ds.get_others():
            logger.info(user)

        logger.info(f"Verified user:")
        logger.info(ds.get_owner())

        owner_data = ds.get_owner_data()
        others_data = ds.get_others_data()

        logger.info(f"Owner data of shape: {owner_data.shape}")
        logger.info(f"Others data  of shape: {others_data.shape}")

if __name__ == '__main__':
    unittest.main()
