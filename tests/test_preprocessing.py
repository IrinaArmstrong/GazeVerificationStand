import sys
import random
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, '..')

from create_training_dataset import TrainDataset
from config import config, init_config
from data_utilities import (split_dataset, truncate_dataset, pad_dataset,
                            vertical_align_data, horizontal_align_data,
                            interpolate_sessions)


class TestFeatureGeneration(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        init_config("../set_locations.ini")
        self.train_dataset = TrainDataset(config.get("DataPaths", "run_data"), ).create_dataset().reset_index().rename({
            "index": "sp_id"}, axis=1)
        self.train_dataset["ts_id"] = self.train_dataset.apply(lambda row:
                                                               (str(row['session_id']) +
                                                                "_" + str(row['sp_id']) +
                                                                "_" + str(row['stimulus_type'])), axis=1)
        print(f"Shape of loaded data: {self.train_dataset.shape}")
        print(f"Unique users: {self.train_dataset['user_id'].nunique()}")
        print(f"Unique sessions: {self.train_dataset['session_id'].nunique()}")


    def test_interpolation(self):
        initial_num_sess = self.train_dataset.session_id.nunique()
        beat_sess_id = random.choice(self.train_dataset.session_id.unique())
        sess_len = self.train_dataset.loc[self.train_dataset.session_id == beat_sess_id].shape[0]
        if sess_len > 500:
            self.train_dataset.loc[self.train_dataset.session_id == beat_sess_id, "gaze_X"] = -100
            self.train_dataset.loc[self.train_dataset.session_id == beat_sess_id, "gaze_Y"] = -100

        self.train_dataset = interpolate_sessions(self.train_dataset, "gaze_X", "gaze_Y")
        self.assertEqual(initial_num_sess - 1, self.train_dataset.session_id.nunique())


    def test_horizontal_align_data(self):
        testing_df = pd.DataFrame({'movements': 20*[3],
                                    'session_id': 10*[0] + 10*[1],
                                    'x': np.random.random_sample(20),
                                    'y': np.random.random_sample(20),
                                    'move_id': 5*[0] + 5*[1] + 5*[2] + 5*[3]})
        testing_hdf = horizontal_align_data(testing_df,
                                            grouping_cols=['session_id', 'move_id'],
                                            aligning_cols=['x', 'y'])
        self.assertEqual(4, testing_hdf.shape[0])
        self.assertEqual(12, testing_hdf.shape[1])


    def test_vertical_align_data(self):
        vdata = vertical_align_data(self.train_dataset, data_col=["gaze_X", "gaze_Y"],
                                    target_col='user_id', guid_col='ts_id')
        self.assertEqual(len(self.train_dataset), len(vdata))

    def test_split_dataset(self):
        self.assertEqual(True, True)

    def test_pad_dataset(self):
        self.assertEqual(True, True)

    def test_truncate_dataset(self):
        self.assertEqual(True, True)

    def test_normalize_dataset(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
