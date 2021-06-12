import os
import sys
import random
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, '..')

from datasets import TrainDataset
from config import config, init_config
from verification.train_dataloaders import create_training_dataloaders



class TestTraining(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        init_config("../set_locations.ini")


    def test_create_dataloaders(self):
        n_rows = 600

        data = pd.DataFrame({
            'data_scaled': [list(np.random.uniform(0, 1, size=(120,))) for _ in range(n_rows)],
            'user_id': [sss for ss in [[s]*60 for s in np.arange(0, n_rows // 60)] for sss in ss],
            'unique_session_id': [sss for ss in [[s] * 10 for s in np.arange(0, n_rows // 10)] for sss in ss]
        })
        dataloaders = create_training_dataloaders(data,
                                                  splitting_params_fn=config.get('Preprocessing',
                                                                                 'processing_params'),
                                                  batching_params_fn=config.get('GazeVerification', 'model_params'))
        self.assertEqual(2, len(dataloaders))
        self.assertIn("train", list(dataloaders.keys()))
        self.assertIn("val", list(dataloaders.keys()))


if __name__ == '__main__':
    unittest.main()
